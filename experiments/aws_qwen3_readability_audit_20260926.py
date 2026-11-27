#!/usr/bin/env python3
"""Run the frozen readability-judge audit with Qwen3 32B on Amazon Bedrock.

Inference happens only in Bedrock. This local process reconstructs and validates
the frozen presentations, sends one blinded text per request, and checkpoints
the returned records. Brown text is never copied into the result file.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import local_llm_readability_audit_20260926 as frozen  # noqa: E402

MODEL_ID = "qwen.qwen3-32b-v1:0"
REGION = "us-east-2"
EXPERIMENT_ID = "aws-qwen3-32b-readability-audit-20260926"
DEFAULT_OUTPUT = ROOT / "runs/aws-qwen3-32b-readability-audit-20260926.json"
DEFAULT_MAX_COST_USD = 2.00
INPUT_USD_PER_1K = 0.000150
OUTPUT_USD_PER_1K = 0.000600
PRICING_EFFECTIVE_DATE = "2026-09-01"
PRICING_SNAPSHOT = ROOT / "experiments/pricing_snapshots/aws-bedrock-qwen3-32b-us-east-2-20260926.json"
PRICING_SKUS = {
    "input_standard": "REQSN4M8AVCDU8BK",
    "output_standard": "SQ4MM597ZHN23FMA",
}
MAX_CONTEXT_TOKENS = 32768
MAX_OUTPUT_TOKENS = 128
CALIBRATION_CONDITIONS = {
    "heldout_brown_prose",
    "heldout_brown_same_word_shuffle",
}

# Qwen3 exposes a model-specific no-thinking switch in its prompt format.
# Keep the frozen rubric verbatim after this serving-mode directive.
SYSTEM_PROMPT = "/no_think\n" + frozen.SYSTEM_PROMPT
USER_PROMPT_TEMPLATE = frozen.USER_PROMPT_TEMPLATE

# Bedrock structured outputs do not support numeric minimum/maximum constraints.
# Enumerating the same four allowed rubric values preserves the requested scale.
BEDROCK_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "grammaticality": {"type": "integer", "enum": [0, 1, 2, 3]},
        "coherent_meaning": {"type": "integer", "enum": [0, 1, 2, 3]},
    },
    "required": ["grammaticality", "coherent_meaning"],
    "additionalProperties": False,
}

DECODING = {"temperature": 0.0, "topP": 1.0, "maxTokens": 128}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def make_runtime_client() -> Any:
    """Reuse AWS CLI's existing short-lived login without printing or saving it."""
    exported = subprocess.run(
        ["aws", "configure", "export-credentials", "--format", "process"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if exported.returncode:
        raise RuntimeError("AWS CLI could not export the existing login credentials")
    credentials = json.loads(exported.stdout)
    required = ("AccessKeyId", "SecretAccessKey", "SessionToken")
    if any(not credentials.get(key) for key in required):
        raise RuntimeError("AWS CLI returned incomplete temporary credentials")

    import boto3
    from botocore.config import Config

    session = boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name=REGION,
    )
    # Disable transparent retries so every attempt and its outcome are explicit
    # in the checkpoint. Resume appends a new attempt without erasing failures.
    return session.client(
        "bedrock-runtime",
        region_name=REGION,
        config=Config(
            connect_timeout=10,
            read_timeout=180,
            retries={"total_max_attempts": 1},
        ),
    )


def public_presentations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [frozen.public_presentation(row) for row in rows]
    if any("text" in row or any(key.endswith("_private") for key in row) for row in result):
        raise ValueError("saved presentation manifest would contain text or private keys")
    return result


def build_schema_request() -> dict[str, Any]:
    return {
        "textFormat": {
            "type": "json_schema",
            "structure": {
                "jsonSchema": {
                    "name": "readability_ratings",
                    "description": "Two independent integer ratings on the frozen 0-3 rubric.",
                    "schema": json.dumps(BEDROCK_OUTPUT_SCHEMA, separators=(",", ":")),
                }
            },
        }
    }


def request_body(text: str, presentation_id: str) -> dict[str, Any]:
    return {
        "modelId": MODEL_ID,
        "system": [{"text": SYSTEM_PROMPT}],
        "messages": [{
            "role": "user",
            "content": [{"text": USER_PROMPT_TEMPLATE.format(text=text)}],
        }],
        "inferenceConfig": DECODING,
        "serviceTier": {"type": "default"},
        "outputConfig": build_schema_request(),
        "requestMetadata": {
            "experiment": EXPERIMENT_ID,
            "presentation": presentation_id,
        },
    }


def request_sha256(text: str, presentation_id: str) -> str:
    body = request_body(text, presentation_id)
    serialized = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(serialized.encode("utf-8"))


def ask_bedrock(client: Any, text: str, presentation_id: str) -> dict[str, Any]:
    body = request_body(text, presentation_id)
    serialized = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    started = time.monotonic()
    response = client.converse(**body)
    latency_seconds = round(time.monotonic() - started, 4)

    message = response.get("output", {}).get("message", {})
    content = message.get("content", [])
    raw_text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    usage = response.get("usage", {})
    input_tokens = usage.get("inputTokens")
    output_tokens = usage.get("outputTokens")
    if type(input_tokens) is int and type(output_tokens) is int:
        cost = (input_tokens * INPUT_USD_PER_1K + output_tokens * OUTPUT_USD_PER_1K) / 1000
        cost = round(cost, 9)
    else:
        cost = None

    parse_error = None
    parsed: dict[str, Any] | None
    try:
        parsed = json.loads(raw_text)
        if set(parsed) != {"grammaticality", "coherent_meaning"}:
            raise ValueError("response does not contain exactly the two rubric fields")
        for name, value in parsed.items():
            if type(value) is not int or value not in (0, 1, 2, 3):
                raise ValueError(f"invalid {name} score: {value!r}")
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        parsed = None
        parse_error = f"{type(exc).__name__}: {exc}"
    if cost is None:
        parse_error = (parse_error + "; " if parse_error else "") + "Bedrock response omitted input/output token usage"

    # Keep the exact generated JSON plus the unmodified response fields relevant
    # to replay and provenance; never save prompt text (especially Brown text).
    metadata = response.get("ResponseMetadata", {})
    safe_response = {
        "output": response.get("output"),
        "stopReason": response.get("stopReason"),
        "usage": usage,
        "metrics": response.get("metrics"),
        "performanceConfig": response.get("performanceConfig"),
        "response_metadata": {
            "request_id": metadata.get("RequestId"),
            "http_status_code": metadata.get("HTTPStatusCode"),
            "retry_attempts": metadata.get("RetryAttempts"),
        },
    }
    return {
        "scores": parsed,
        "parse_error": parse_error,
        "raw_json": raw_text,
        "raw_response": safe_response,
        "request_sha256": sha256_bytes(serialized.encode("utf-8")),
        "text_sha256": frozen.sha256_text(text),
        "aws_request_id": metadata.get("RequestId"),
        "http_status_code": metadata.get("HTTPStatusCode"),
        "api_metrics": {
            "input_tokens": input_tokens if type(input_tokens) is int else None,
            "output_tokens": output_tokens if type(output_tokens) is int else None,
            "total_tokens": usage.get("totalTokens"),
            "latency_seconds": latency_seconds,
            "estimated_cost_usd": cost,
            "retry_attempts": metadata.get("RetryAttempts"),
        },
        "request_completed_utc": utc_now(),
    }


def collect_runtime_facts() -> dict[str, Any]:
    identity_proc = subprocess.run(
        ["aws", "sts", "get-caller-identity", "--output", "json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if identity_proc.returncode:
        raise RuntimeError("could not verify the active AWS identity before model inference")
    identity = json.loads(identity_proc.stdout)
    arn = identity.get("Arn", "")
    principal_type = "account-root" if arn.endswith(":root") else (
        "assumed-role" if ":assumed-role/" in arn else "iam-user" if ":user/" in arn else "other"
    )
    version_proc = subprocess.run(
        ["aws", "--version"], check=False, capture_output=True, text=True, timeout=10
    )
    import boto3
    import botocore

    return {
        "credential_source": "AWS CLI temporary credentials; values not written to the run",
        "active_profile": os.environ.get("AWS_PROFILE", "default"),
        "caller_principal_type": principal_type,
        "account_id_sha256_prefix": sha256_bytes(identity.get("Account", "").encode())[:16],
        "caller_arn_sha256_prefix": sha256_bytes(arn.encode())[:16],
        "aws_cli_version": version_proc.stdout.strip() or version_proc.stderr.strip(),
        "boto3_version": boto3.__version__,
        "botocore_version": botocore.__version__,
        "sdk_retry_policy": "total_max_attempts=1; no transparent retries",
    }


def runtime_identity_key(facts: dict[str, Any]) -> tuple[str, str, str, str]:
    """Bind a resumable run to the same AWS profile and caller identity."""
    return (
        facts.get("active_profile", ""),
        facts.get("account_id_sha256_prefix", ""),
        facts.get("caller_arn_sha256_prefix", ""),
        facts.get("caller_principal_type", ""),
    )


def same_runtime_identity(recorded: dict[str, Any], current: dict[str, Any]) -> bool:
    return runtime_identity_key(recorded) == runtime_identity_key(current)


def max_request_cost_usd() -> float:
    """Conservative per-call bound from the documented context and output caps."""
    return (
        MAX_CONTEXT_TOKENS * INPUT_USD_PER_1K
        + MAX_OUTPUT_TOKENS * OUTPUT_USD_PER_1K
    ) / 1000


def execute_one(
    client: Any,
    row: dict[str, Any],
    stage: str,
    result: dict[str, Any],
    responses: list[dict[str, Any]],
    output_path: Path,
    current_budget: float,
    max_cost: float,
) -> float:
    presentation_id = row["presentation_id"]
    reserved_upper_bound = max_request_cost_usd()
    if current_budget + reserved_upper_bound > max_cost:
        result["status"] = "cost_cap_exceeded"
        record_cost_fields(result, current_budget)
        atomic_write(output_path, result)
        raise RuntimeError("run budget cannot reserve the worst-case cost of one more request")

    failures = [
        attempt for attempt in result.get("failed_attempts", [])
        if attempt.get("presentation_id") == presentation_id
    ]
    in_flight = {
        "presentation_id": presentation_id,
        "stage": stage,
        "attempt_number": len(failures) + 1,
        "request_sha256": request_sha256(row["text"], presentation_id),
        "text_sha256": frozen.sha256_text(row["text"]),
        "started_utc": utc_now(),
        "status": "in_flight",
        "billing_status": "unknown until response is checkpointed",
        "reserved_max_cost_usd": round(reserved_upper_bound, 9),
    }
    result["in_flight_attempt"] = in_flight
    result["status"] = f"running_{stage}"
    record_cost_fields(result, current_budget + reserved_upper_bound)
    atomic_write(output_path, result)
    try:
        response = ask_bedrock(client, row["text"], presentation_id)
    except Exception as exc:
        in_flight["status"] = "ambiguous"
        in_flight["billing_status"] = "unknown; explicit retry required"
        in_flight["error_type"] = type(exc).__name__
        in_flight["error"] = "Call ended without a confirmed response; billing is conservatively reserved."
        in_flight["finished_utc"] = utc_now()
        in_flight["budget_charge_usd"] = reserved_upper_bound
        result["status"] = "ambiguous_model_call"
        record_cost_fields(result, current_budget + reserved_upper_bound)
        atomic_write(output_path, result)
        raise RuntimeError(
            "Bedrock did not return a confirmed response; run is checkpointed and will not retry without --retry-failed-attempts"
        ) from exc

    response["presentation_id"] = presentation_id
    response["condition_hidden"] = True
    response["attempt_number"] = in_flight["attempt_number"]
    cost = response["api_metrics"].get("estimated_cost_usd")
    budget_charge = reserved_upper_bound
    current_budget += budget_charge
    in_flight["finished_utc"] = utc_now()
    in_flight["budget_charge_usd"] = budget_charge
    response["budget_charge_usd"] = budget_charge
    response["budget_charge_basis"] = "per-request worst-case token bound"
    result["in_flight_attempt"] = None
    if response["parse_error"]:
        brown_text = row["condition"] in CALIBRATION_CONDITIONS
        failure = {
            **in_flight,
            "status": "response_received_invalid",
            "billing_status": "token-derived estimate recorded" if type(cost) in (float, int) else "unknown; worst-case budget bound used",
            "error_type": "InvalidModelOutput",
            "aws_request_id": response["aws_request_id"],
            "http_status_code": response["http_status_code"],
            "api_metrics": response["api_metrics"],
            "estimated_cost_usd": cost,
            "budget_charge_usd": budget_charge,
            "raw_json_sha256": sha256_bytes(response["raw_json"].encode("utf-8")),
            "raw_json_length": len(response["raw_json"]),
        }
        if brown_text:
            # Never persist licensed Brown text, even if a malformed response
            # happens to echo its input in the JSON or response envelope.
            failure["error"] = "Model output did not match the frozen score schema."
        else:
            failure.update({
                "error": response["parse_error"],
                "raw_json": response["raw_json"],
                "raw_response": response["raw_response"],
            })
        result.setdefault("failed_attempts", []).append({
            "finished_utc": utc_now(),
            **failure,
        })
        result["status"] = "failed_partial"
        record_cost_fields(result, current_budget)
        atomic_write(output_path, result)
        raise RuntimeError("Bedrock returned a response that failed the frozen score schema")

    response.pop("parse_error", None)
    responses.append(response)
    result["responses"] = responses
    record_cost_fields(result, current_budget)
    result["status"] = "running_calibration" if stage == "calibration" else "running_candidates"
    atomic_write(output_path, result)
    return current_budget


def actual_cost_accounted(result: dict[str, Any]) -> float:
    response_cost = sum(
        row.get("api_metrics", {}).get("estimated_cost_usd") or 0
        for row in result.get("responses", [])
    )
    failed_cost = sum(
        row.get("estimated_cost_usd") or 0
        for row in result.get("failed_attempts", [])
    )
    return response_cost + failed_cost


def record_cost_fields(result: dict[str, Any], budget_reserved: float) -> None:
    known = actual_cost_accounted(result)
    missing_response_usage = any(
        row.get("api_metrics", {}).get("estimated_cost_usd") is None
        for row in result.get("responses", [])
    )
    missing_failed_usage = any(
        row.get("estimated_cost_usd") is None
        for row in result.get("failed_attempts", [])
    )
    unknown_in_flight = result.get("in_flight_attempt") is not None
    complete = not (missing_response_usage or missing_failed_usage or unknown_in_flight)
    result["estimated_cost_usd"] = round(known, 9) if complete else None
    result["known_token_cost_subtotal_usd"] = round(known, 9)
    result["actual_cost_estimate_complete"] = complete
    result["budget_reserved_usd"] = round(budget_reserved, 9)


def cost_accounted(result: dict[str, Any]) -> float:
    """Worst-case budget reserved by every successful and failed request."""
    response_cost = sum(
        row.get("budget_charge_usd", max_request_cost_usd())
        for row in result.get("responses", [])
    )
    failed_cost = sum(
        row.get("budget_charge_usd", row.get("estimated_cost_usd", max_request_cost_usd()))
        for row in result.get("failed_attempts", [])
    )
    in_flight = result.get("in_flight_attempt") or {}
    in_flight_cost = in_flight.get("budget_charge_usd", in_flight.get("reserved_max_cost_usd", 0)) or 0
    return response_cost + failed_cost + in_flight_cost


def checkpoint_interrupted_attempt(result: dict[str, Any]) -> bool:
    """Turn a crash-left in-flight marker into an auditable ambiguous attempt."""
    attempt = result.get("in_flight_attempt")
    if not attempt:
        return False
    attempt["status"] = "interrupted_unknown"
    attempt["billing_status"] = "unknown after process interruption; explicit retry required"
    attempt["finished_utc"] = utc_now()
    attempt["budget_charge_usd"] = attempt.get("reserved_max_cost_usd", max_request_cost_usd())
    result.setdefault("failed_attempts", []).append(attempt)
    result["in_flight_attempt"] = None
    result["status"] = "awaiting_explicit_retry"
    return True


def calibration_summary(rows: list[dict[str, Any]], responses: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["presentation_id"]: row for row in rows}
    scored = {row["presentation_id"]: row for row in responses}
    positives = []
    shuffles = []
    for row in rows:
        if row["presentation_id"] not in scored:
            continue
        response = scored[row["presentation_id"]]
        scores = response["scores"]
        passed = int(scores["grammaticality"] >= 2 and scores["coherent_meaning"] >= 2)
        item = {"presentation_id": row["presentation_id"], "joint_pass": passed}
        if row["condition"] == "heldout_brown_prose":
            positives.append(item)
        elif row["condition"] == "heldout_brown_same_word_shuffle":
            shuffles.append(item)
    # Complete only when all 24 controls have been rated.
    complete = len(positives) == 12 and len(shuffles) == 12
    positive_passes = sum(row["joint_pass"] for row in positives)
    shuffle_passes = sum(row["joint_pass"] for row in shuffles)
    passes_gate = complete and positive_passes >= 8 and shuffle_passes <= 3
    by_source: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        if row["condition"] not in CALIBRATION_CONDITIONS:
            continue
        by_source[row["pair_id"]][row["condition"]] = row
    sentence_counts = sorted(
        row["brown_sentence_indices_half_open"][1] - row["brown_sentence_indices_half_open"][0]
        for pair in by_source.values()
        for row in pair.values()
        if row.get("condition") == "heldout_brown_prose"
    )
    return {
        "complete": complete,
        "brown_prose_joint_passes": positive_passes,
        "brown_prose_n": len(positives),
        "brown_shuffle_joint_passes": shuffle_passes,
        "brown_shuffle_n": len(shuffles),
        "gate_passes": passes_gate,
        "prespecified_gate": "at least 8/12 Brown prose pass both >=2, and at most 3/12 Brown shuffles pass both >=2",
        "candidate_results_released": passes_gate,
        "brown_sentences_per_positive_control": sentence_counts,
        "multi_sentence_positive_controls": sum(count > 1 for count in sentence_counts),
        "interpretation": (
            "basic known-group sensitivity check for this model and these controls; not human validation"
            if passes_gate else "candidate scores are inconclusive under the prespecified judge-calibration gate"
        ),
    }


def build_fingerprint(
    rows: list[dict[str, Any]],
    input_hashes: dict[str, str],
    brown_manifest_hash: str,
    script_hash: str,
    frozen_script_hash: str,
    pricing_snapshot_hash: str,
    max_cost: float,
) -> str:
    prompt_hash = sha256_bytes((SYSTEM_PROMPT + "\0" + USER_PROMPT_TEMPLATE).encode("utf-8"))
    material = {
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "region": REGION,
        "api": "Amazon Bedrock Runtime Converse",
        "input_hashes": input_hashes,
        "brown_control_manifest_sha256": brown_manifest_hash,
        "scorer_script_sha256": script_hash,
        "frozen_protocol_script_sha256": frozen_script_hash,
        "pricing_snapshot_sha256": pricing_snapshot_hash,
        "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
        "user_prompt_template_sha256": sha256_bytes(USER_PROMPT_TEMPLATE.encode("utf-8")),
        "prompt_sha256": prompt_hash,
        "decoding": DECODING,
        "service_tier": "default",
        "response_schema": BEDROCK_OUTPUT_SCHEMA,
        "presentation_manifest": public_presentations(rows),
        "control_gate": {"positive_minimum": 8, "shuffle_maximum": 3, "n_each": 12},
        "max_cost_usd": max_cost,
        "pricing": {
            "input_usd_per_1k": INPUT_USD_PER_1K,
            "output_usd_per_1k": OUTPUT_USD_PER_1K,
            "effective_date": PRICING_EFFECTIVE_DATE,
        },
    }
    return sha256_bytes(json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-failed-attempts",
        action="store_true",
        help="explicitly permit new billable requests for inputs with an ambiguous/invalid prior attempt",
    )
    parser.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST_USD)
    args = parser.parse_args()
    if not 0 < args.max_cost_usd <= 10:
        parser.error("--max-cost-usd must be greater than 0 and no more than $10")
    if args.retry_failed_attempts and not args.resume:
        parser.error("--retry-failed-attempts requires --resume")

    candidates, input_hashes = frozen.load_candidate_rows(
        frozen.INPUT_CANDIDATES, frozen.INPUT_STATIC
    )
    brown_controls, brown_manifest_hash, brown_source = frozen.load_brown_controls(
        frozen.INPUT_LENGTH_CONTROLS, candidates
    )
    rows = frozen.make_presentations(candidates, brown_controls)
    calibration_rows = [row for row in rows if row["condition"] in CALIBRATION_CONDITIONS]
    remaining_rows = [row for row in rows if row["condition"] not in CALIBRATION_CONDITIONS]
    if (len(rows), len(calibration_rows), len(remaining_rows)) != (244, 24, 220):
        raise ValueError("unexpected frozen audit presentation counts")

    if args.prepare_only:
        print(json.dumps({
            "model_id": MODEL_ID,
            "region": REGION,
            "total_presentations": len(rows),
            "calibration_first": len(calibration_rows),
            "candidate_and_shuffle_presentations": len(remaining_rows),
            "candidate_count": len(candidates),
            "brown_positive_controls_with_multiple_sentences": sum(
                row["condition"] == "heldout_brown_prose"
                and row["brown_sentence_indices_half_open"][1] - row["brown_sentence_indices_half_open"][0] > 1
                for row in rows
            ),
            "presentation_manifest_sha256": sha256_bytes(json.dumps(
                public_presentations(rows), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")),
            "pricing_snapshot_sha256": sha256_bytes(PRICING_SNAPSHOT.read_bytes()),
        }, indent=2))
        return 0

    script_hash = sha256_bytes(Path(__file__).read_bytes())
    frozen_script_hash = sha256_bytes(Path(frozen.__file__).read_bytes())
    pricing_snapshot_hash = sha256_bytes(PRICING_SNAPSHOT.read_bytes())
    run_fingerprint = build_fingerprint(
        rows, input_hashes, brown_manifest_hash, script_hash, frozen_script_hash,
        pricing_snapshot_hash, args.max_cost_usd,
    )
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"{args.output} exists; use a new path or --resume")
    if args.resume:
        if not args.output.exists():
            raise FileNotFoundError("--resume requested but output does not exist")
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if result.get("run_fingerprint") != run_fingerprint:
            raise ValueError("saved run fingerprint differs from frozen inputs/model/prompt/settings")
        if result.get("status") == "calibration_failed":
            print(json.dumps(result["calibration"], indent=2))
            return 2
        if result.get("status") == "complete":
            print(json.dumps(result.get("analysis", {}), indent=2, ensure_ascii=False))
            return 0
        responses = result.get("responses", [])
        checkpoint_interrupted_attempt(result)
        record_cost_fields(result, cost_accounted(result))
    else:
        result = {
            "experiment_id": EXPERIMENT_ID,
            "status": "running_calibration",
            "started_utc": utc_now(),
            "run_fingerprint": run_fingerprint,
            "source_artifacts": {
                "candidate_packet": frozen.INPUT_CANDIDATES.relative_to(ROOT).as_posix(),
                "static_verifier_input": frozen.INPUT_STATIC.relative_to(ROOT).as_posix(),
                "heldout_brown_control_manifest": frozen.INPUT_LENGTH_CONTROLS.relative_to(ROOT).as_posix(),
                "local_protocol_script": Path(frozen.__file__).relative_to(ROOT).as_posix(),
                "aws_scorer_script": Path(__file__).relative_to(ROOT).as_posix(),
                "aws_scorer_script_sha256": script_hash,
                "frozen_protocol_script": Path(frozen.__file__).relative_to(ROOT).as_posix(),
                "frozen_protocol_script_sha256": frozen_script_hash,
                "pricing_snapshot": PRICING_SNAPSHOT.relative_to(ROOT).as_posix(),
                "pricing_snapshot_sha256": pricing_snapshot_hash,
                **input_hashes,
                "heldout_brown_control_manifest_sha256": brown_manifest_hash,
            },
            "model_facts": {
                "provider": "Alibaba Qwen via Amazon Bedrock",
                "model_id": MODEL_ID,
                "region": REGION,
                "api": "bedrock-runtime Converse",
                "model_weight_digest": None,
                "model_weight_digest_note": "not exposed for this Bedrock-managed model ID",
                "reasoning_mode": {
                    "requested": "Qwen3 /no_think prompt directive",
                    "enforcement": "soft prompt-level directive; Bedrock does not expose a separate reasoning-mode control here",
                },
                "endpoint_provisioned": False,
            },
            "pricing": {
                "request_service_tier": "default",
                "pricing_tier": "standard on-demand",
                "input_usd_per_1000_tokens": INPUT_USD_PER_1K,
                "output_usd_per_1000_tokens": OUTPUT_USD_PER_1K,
                "effective_date": PRICING_EFFECTIVE_DATE,
                "source": "AWS Price List API, AmazonBedrock, Qwen3 32B, us-east-2, standard on-demand",
                "price_list_query": {
                    "api": "GetProducts",
                    "endpoint_region": "us-east-1",
                    "filters": {"location": "US East (Ohio)", "model": "Qwen3 32B"},
                    "effective_date": PRICING_EFFECTIVE_DATE,
                    "retrieved_on": "2026-09-26",
                    "standard_input_sku": PRICING_SKUS["input_standard"],
                    "standard_output_sku": PRICING_SKUS["output_standard"],
                    "snapshot_sha256": pricing_snapshot_hash,
                },
                "source_url": "https://aws.amazon.com/bedrock/pricing/",
                "run_cost_cap_usd": args.max_cost_usd,
                "per_request_worst_case_usd": round(max_request_cost_usd(), 9),
                "full_244_request_worst_case_usd": round(244 * max_request_cost_usd(), 6),
                "budget_accounting": "Every attempt reserves the full per-request token bound, including successful calls, so resumed runs and explicit retries cannot forget prior spend. estimated_cost_usd is the token-usage cost estimate only when usage is complete; otherwise it is null and known_token_cost_subtotal_usd reports the known subtotal.",
            },
            "method": {
                "model_blind_to_condition": True,
                "one_text_per_request": True,
                "condition_labels_and_ids_in_prompt": False,
                "calibration_first": True,
                "candidate_scoring_stops_if_calibration_fails": True,
                "candidate_count": 52,
                "candidate_same_word_shuffles_per_text": 3,
                "brown_positive_controls": 12,
                "brown_same_word_shuffle_controls": 12,
                "brown_source": brown_source,
                "prompt_source": "frozen Qwen2.5 readability rubric; Qwen3 adds only the documented /no_think serving-mode directive",
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt_template": USER_PROMPT_TEMPLATE,
                "decoding": DECODING,
                "seed": None,
                "seed_note": "Bedrock Converse common inferenceConfig has no seed parameter; presentation order and shuffles are seeded; 12 hidden repeats measure response stability.",
                "response_schema": BEDROCK_OUTPUT_SCHEMA,
                "response_schema_note": "Bedrock structured output uses enum 0-3 because its schema subset omits numeric minimum/maximum; score space is unchanged.",
                "calibration_gate": "at least 8/12 intact Brown controls pass both dimensions >=2 and at most 3/12 Brown shuffles pass both >=2",
                "primary_outcome": "candidate joint score >=2 on both dimensions; paired against mean of three matched word shuffles",
                "limitations": [
                    "LLM judgments are not human ratings or validated readability measurements.",
                    "Brown controls are not human-validated gold labels; model pretraining exposure is unknown.",
                    "The frozen Brown positive set includes multi-sentence spans; report their count and retain the prespecified gate.",
                    "A gate pass is basic control sensitivity for this model and these controls only.",
                ],
            },
            "presentations": public_presentations(rows),
            "responses": [],
            "failed_attempts": [],
            "in_flight_attempt": None,
        }
        responses = []

    if args.resume:
        result.pop("pending_retry_count", None)
        completed_before_retry = {row["presentation_id"] for row in responses}
        pending_failures = [
            attempt for attempt in result.get("failed_attempts", [])
            if attempt.get("presentation_id") not in completed_before_retry
        ]
        if pending_failures and not args.retry_failed_attempts:
            result["status"] = "awaiting_explicit_retry"
            result["pending_retry_count"] = len(pending_failures)
            record_cost_fields(result, cost_accounted(result))
            atomic_write(args.output, result)
            print(
                f"{len(pending_failures)} unresolved/invalid presentation(s); resume again with --retry-failed-attempts to authorize another billable attempt",
                file=sys.stderr,
            )
            return 4

    runtime_facts = collect_runtime_facts()
    if args.resume:
        recorded_facts = result.get("runtime_environment")
        if not recorded_facts:
            result["status"] = "runtime_identity_unverifiable"
            atomic_write(args.output, result)
            print("saved run has no original AWS identity record; refusing to resume", file=sys.stderr)
            return 5
        result.setdefault("runtime_environment_checks", []).append({
            "checked_utc": utc_now(),
            **runtime_facts,
        })
        if not same_runtime_identity(recorded_facts, runtime_facts):
            result["status"] = "runtime_identity_mismatch"
            record_cost_fields(result, cost_accounted(result))
            atomic_write(args.output, result)
            print("active AWS profile/account/principal differs from the original run; refusing to resume", file=sys.stderr)
            return 5
    else:
        result["runtime_environment"] = runtime_facts
        result["runtime_environment_checks"] = [{"checked_utc": utc_now(), **runtime_facts}]
    atomic_write(args.output, result)
    client = make_runtime_client()
    completed = {row["presentation_id"] for row in responses}
    budget_reserved = cost_accounted(result)
    record_cost_fields(result, budget_reserved)
    if budget_reserved + 1e-12 >= args.max_cost_usd:
        result["status"] = "cost_cap_exceeded"
        atomic_write(args.output, result)
        print(f"run budget cap reached; no further model calls made (${budget_reserved:.4f} reserved)", file=sys.stderr)
        return 3
    result["status"] = "running_calibration"
    atomic_write(args.output, result)
    started = time.monotonic()
    try:
        for row in calibration_rows:
            presentation_id = row["presentation_id"]
            if presentation_id in completed:
                continue
            budget_reserved = execute_one(
                client, row, "calibration", result, responses, args.output,
                budget_reserved, args.max_cost_usd,
            )
            completed.add(presentation_id)
            print(f"calibration {sum(r['presentation_id'] in completed for r in calibration_rows)}/24; known token-cost subtotal ${actual_cost_accounted(result):.4f}; worst-case budget ${budget_reserved:.4f}", flush=True)

        calibration = calibration_summary(rows, responses)
        result["calibration"] = calibration
        if not calibration["gate_passes"]:
            result["status"] = "calibration_failed"
            result["finished_utc"] = utc_now()
            result["elapsed_seconds"] = round(time.monotonic() - started, 3)
            record_cost_fields(result, budget_reserved)
            atomic_write(args.output, result)
            print(json.dumps(calibration, indent=2))
            return 2

        result["status"] = "running_candidates"
        atomic_write(args.output, result)
        for index, row in enumerate(remaining_rows, start=1):
            presentation_id = row["presentation_id"]
            if presentation_id in completed:
                continue
            budget_reserved = execute_one(
                client, row, "candidate_and_shuffle", result, responses, args.output,
                budget_reserved, args.max_cost_usd,
            )
            completed.add(presentation_id)
            print(f"candidate/control {index}/220; known token-cost subtotal ${actual_cost_accounted(result):.4f}; worst-case budget ${budget_reserved:.4f}", flush=True)

        result["analysis"] = frozen.compute_analysis(rows, responses)
        result["status"] = "complete"
        result["finished_utc"] = utc_now()
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        record_cost_fields(result, budget_reserved)
        atomic_write(args.output, result)
        print(json.dumps(result["analysis"], indent=2, ensure_ascii=False))
        return 0
    except Exception:
        result["responses"] = responses
        record_cost_fields(result, cost_accounted(result))
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        atomic_write(args.output, result)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
