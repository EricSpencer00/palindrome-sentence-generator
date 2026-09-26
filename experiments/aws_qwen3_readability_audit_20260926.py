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
DEFAULT_MAX_COST_USD = 1.00
INPUT_USD_PER_1K = 0.000150
OUTPUT_USD_PER_1K = 0.000600
PRICING_EFFECTIVE_DATE = "2026-09-01"
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
        "outputConfig": build_schema_request(),
        "requestMetadata": {
            "experiment": EXPERIMENT_ID,
            "presentation": presentation_id,
        },
    }


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
    if type(input_tokens) is not int or type(output_tokens) is not int:
        raise ValueError("Bedrock response omitted input/output token usage")
    cost = (input_tokens * INPUT_USD_PER_1K + output_tokens * OUTPUT_USD_PER_1K) / 1000

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

    # Keep the exact generated JSON plus the unmodified response fields relevant
    # to replay and provenance; never save prompt text (especially Brown text).
    safe_response = {
        "output": response.get("output"),
        "stopReason": response.get("stopReason"),
        "usage": usage,
        "metrics": response.get("metrics"),
        "performanceConfig": response.get("performanceConfig"),
    }
    metadata = response.get("ResponseMetadata", {})
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
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": usage.get("totalTokens"),
            "latency_seconds": latency_seconds,
            "estimated_cost_usd": round(cost, 9),
        },
        "request_completed_utc": utc_now(),
    }


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
        "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
        "user_prompt_template_sha256": sha256_bytes(USER_PROMPT_TEMPLATE.encode("utf-8")),
        "prompt_sha256": prompt_hash,
        "decoding": DECODING,
        "response_schema": BEDROCK_OUTPUT_SCHEMA,
        "presentation_manifest": public_presentations(rows),
        "control_gate": {"positive_minimum": 8, "shuffle_maximum": 3, "n_each": 12},
        "max_cost_usd": max_cost,
    }
    return sha256_bytes(json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST_USD)
    args = parser.parse_args()
    if not 0 < args.max_cost_usd <= 10:
        parser.error("--max-cost-usd must be greater than 0 and no more than $10")

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
        }, indent=2))
        return 0

    script_hash = sha256_bytes(Path(__file__).read_bytes())
    run_fingerprint = build_fingerprint(
        rows, input_hashes, brown_manifest_hash, script_hash, args.max_cost_usd
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
        responses = result.get("responses", [])
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
            "reasoning_mode": "Qwen3 /no_think directive; no rationale requested or returned",
                "endpoint_provisioned": False,
            },
            "pricing": {
                "service_tier": "standard",
                "input_usd_per_1000_tokens": INPUT_USD_PER_1K,
                "output_usd_per_1000_tokens": OUTPUT_USD_PER_1K,
                "effective_date": PRICING_EFFECTIVE_DATE,
                "source": "AWS Price List API, AmazonBedrock, Qwen3 32B, us-east-2, standard on-demand",
                "source_url": "https://aws.amazon.com/bedrock/pricing/",
                "run_cost_cap_usd": args.max_cost_usd,
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
        }
        responses = []

    client = make_runtime_client()
    completed = {row["presentation_id"] for row in responses}
    total_cost = sum(row.get("api_metrics", {}).get("estimated_cost_usd", 0) for row in responses)
    total_cost += sum(row.get("estimated_cost_usd", 0) or 0 for row in result.get("failed_attempts", []))
    if total_cost >= args.max_cost_usd:
        result["status"] = "cost_cap_exceeded"
        result["estimated_cost_usd"] = round(total_cost, 9)
        atomic_write(args.output, result)
        print(f"run cost cap reached; no further model calls made (${total_cost:.4f})", file=sys.stderr)
        return 3
    result["status"] = "running_calibration"
    atomic_write(args.output, result)
    started = time.monotonic()
    try:
        for stage_name, stage_rows in (("calibration", calibration_rows),):
            for row in stage_rows:
                presentation_id = row["presentation_id"]
                if presentation_id in completed:
                    continue
                try:
                    response = ask_bedrock(client, row["text"], presentation_id)
                except Exception as exc:
                    result["failed_attempts"].append({
                        "presentation_id": presentation_id,
                        "stage": stage_name,
                        "attempted_utc": utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    })
                    result["responses"] = responses
                    result["status"] = "failed_partial"
                    atomic_write(args.output, result)
                    raise
                total_cost += response["api_metrics"]["estimated_cost_usd"]
                response["presentation_id"] = presentation_id
                response["condition_hidden"] = True
                if response["parse_error"]:
                    result["failed_attempts"].append({
                        "presentation_id": presentation_id,
                        "stage": stage_name,
                        "attempted_utc": utc_now(),
                        "error_type": "InvalidModelOutput",
                        "error": response["parse_error"],
                        "aws_request_id": response["aws_request_id"],
                        "api_metrics": response["api_metrics"],
                        "estimated_cost_usd": response["api_metrics"]["estimated_cost_usd"],
                        "raw_json": response["raw_json"],
                        "raw_response": response["raw_response"],
                    })
                    result["responses"] = responses
                    result["estimated_cost_usd"] = round(total_cost, 9)
                    result["status"] = "failed_partial"
                    atomic_write(args.output, result)
                    raise RuntimeError("Bedrock returned a response that failed the frozen score schema")
                response.pop("parse_error", None)
                responses.append(response)
                completed.add(presentation_id)
                result["responses"] = responses
                result["estimated_cost_usd"] = round(total_cost, 9)
                atomic_write(args.output, result)
                print(f"calibration {len(completed)}/24; estimated cost ${total_cost:.4f}", flush=True)
                if total_cost > args.max_cost_usd:
                    result["status"] = "cost_cap_exceeded"
                    atomic_write(args.output, result)
                    raise RuntimeError("estimated token cost exceeded the configured run cap")

        calibration = calibration_summary(rows, responses)
        result["calibration"] = calibration
        if not calibration["gate_passes"]:
            result["status"] = "calibration_failed"
            result["finished_utc"] = utc_now()
            result["elapsed_seconds"] = round(time.monotonic() - started, 3)
            result["estimated_cost_usd"] = round(total_cost, 9)
            atomic_write(args.output, result)
            print(json.dumps(calibration, indent=2))
            return 2

        result["status"] = "running_candidates"
        atomic_write(args.output, result)
        for index, row in enumerate(remaining_rows, start=1):
            presentation_id = row["presentation_id"]
            if presentation_id in completed:
                continue
            try:
                response = ask_bedrock(client, row["text"], presentation_id)
            except Exception as exc:
                result["failed_attempts"].append({
                    "presentation_id": presentation_id,
                    "stage": "candidate_and_shuffle",
                    "attempted_utc": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                })
                result["responses"] = responses
                result["status"] = "failed_partial"
                atomic_write(args.output, result)
                raise
            total_cost += response["api_metrics"]["estimated_cost_usd"]
            response["presentation_id"] = presentation_id
            response["condition_hidden"] = True
            if response["parse_error"]:
                result["failed_attempts"].append({
                    "presentation_id": presentation_id,
                    "stage": "candidate_and_shuffle",
                    "attempted_utc": utc_now(),
                    "error_type": "InvalidModelOutput",
                    "error": response["parse_error"],
                    "aws_request_id": response["aws_request_id"],
                    "api_metrics": response["api_metrics"],
                    "estimated_cost_usd": response["api_metrics"]["estimated_cost_usd"],
                    "raw_json": response["raw_json"],
                    "raw_response": response["raw_response"],
                })
                result["responses"] = responses
                result["estimated_cost_usd"] = round(total_cost, 9)
                result["status"] = "failed_partial"
                atomic_write(args.output, result)
                raise RuntimeError("Bedrock returned a response that failed the frozen score schema")
            response.pop("parse_error", None)
            responses.append(response)
            completed.add(presentation_id)
            result["responses"] = responses
            result["estimated_cost_usd"] = round(total_cost, 9)
            atomic_write(args.output, result)
            print(f"candidate/control {index}/220; estimated cost ${total_cost:.4f}", flush=True)
            if total_cost > args.max_cost_usd:
                result["status"] = "cost_cap_exceeded"
                atomic_write(args.output, result)
                raise RuntimeError("estimated token cost exceeded the configured run cap")

        result["analysis"] = frozen.compute_analysis(rows, responses)
        result["status"] = "complete"
        result["finished_utc"] = utc_now()
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["estimated_cost_usd"] = round(total_cost, 9)
        atomic_write(args.output, result)
        print(json.dumps(result["analysis"], indent=2, ensure_ascii=False))
        return 0
    except Exception:
        result["responses"] = responses
        result["estimated_cost_usd"] = round(total_cost, 9)
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        atomic_write(args.output, result)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
