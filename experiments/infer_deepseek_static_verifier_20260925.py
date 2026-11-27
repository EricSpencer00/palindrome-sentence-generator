"""Run cost-capped, counterbalanced DeepSeek V3.2 pairwise comparisons via Bedrock."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

MODEL_ID = "deepseek.v3.2"
REGION = "us-east-2"
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 256
INPUT_USD_PER_1K = 0.00062
OUTPUT_USD_PER_1K = 0.00185
DEFAULT_BUDGET_USD = 0.02
DEFAULT_MIN_INTERVAL_SECONDS = 0.25
SYSTEM_PROMPT = (
    "You are comparing two short English texts for ordinary-language quality. "
    "Judge sentence-level grammaticality, natural phrasing, and whether the text "
    "communicates a coherent meaning. Ignore length. One may be deliberately "
    "word-shuffled. Do not infer or reward palindrome status. Do not provide "
    "chain-of-thought. Return one JSON object only with choice A, B, or tie and "
    "a brief observable language feature (at most 12 words)."
)


def parse_answer(text: str) -> dict:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.I)
    match = re.search(r"\{[^{}]*\}", cleaned, re.S)
    if not match:
        return {"choice": None, "observable_feature": None}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"choice": None, "observable_feature": None}
    choice = value.get("choice")
    feature = value.get("observable_feature", value.get("feature", value.get("reason")))
    choice = choice.upper() if isinstance(choice, str) else None
    if choice not in {"A", "B", "TIE"}:
        choice = None
    if isinstance(feature, str):
        feature = " ".join(feature.split())[:240]
    else:
        feature = None
    return {"choice": choice, "observable_feature": feature}


def comparison_specs(input_data: dict) -> list[dict]:
    by_id = {item["id"]: item for item in input_data["items"]}
    specs = []
    for item in input_data["items"]:
        if item["kind"] != "candidate":
            continue
        pair_id = item["pair_id"]
        for baseline_kind, baseline_id in (
            ("candidate_same_word_shuffle", f"shuffle:candidate:{pair_id}:00"),
        ):
            baseline = by_id[baseline_id]
            comparison_id = f"{pair_id}::vs::{baseline_kind}"
            side_hash = hashlib.sha256(f"{input_data['experiment']}:{comparison_id}:sides".encode()).digest()
            candidate_side = "A" if side_hash[0] % 2 == 0 else "B"
            a, b = (item, baseline) if candidate_side == "A" else (baseline, item)
            prompt = (
                "Compare the two passages. Which reads as better ordinary English prose?\n\n"
                f"A: {a['text']}\n\nB: {b['text']}\n\n"
                'Return exactly: {"choice":"A|B|tie","observable_feature":"brief concrete feature"}'
            )
            specs.append({
                "comparison_id": comparison_id,
                "candidate_id": item["id"],
                "baseline_id": baseline_id,
                "baseline_kind": baseline_kind,
                "candidate_side": candidate_side,
                "a_id": a["id"],
                "b_id": b["id"],
                "prompt": prompt,
            })
    return specs


def upper_bound_cost(specs: list[dict]) -> float:
    # Count every UTF-8 byte as an input token and reserve the full output
    # allowance. This deliberately overestimates the short ASCII prompts.
    input_token_bound = sum(len(spec["prompt"].encode("utf-8")) + len(SYSTEM_PROMPT.encode("utf-8")) for spec in specs)
    return input_token_bound / 1000 * INPUT_USD_PER_1K + len(specs) * MAX_TOKENS / 1000 * OUTPUT_USD_PER_1K


def run(input_path: Path, output_path: Path, budget_usd: float = DEFAULT_BUDGET_USD,
        interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
        max_new_comparisons: int | None = None) -> dict:
    import boto3
    from botocore.config import Config

    input_data = json.loads(input_path.read_text())
    specs = comparison_specs(input_data)
    cost_bound = upper_bound_cost(specs)
    if cost_bound > budget_usd:
        raise ValueError(f"worst-case call bound ${cost_bound:.4f} exceeds budget ${budget_usd:.4f}")
    if max_new_comparisons is not None and max_new_comparisons < 1:
        raise ValueError("max_new_comparisons must be positive")

    prior = None
    if output_path.exists():
        prior = json.loads(output_path.read_text())
        if prior.get("input_manifest_sha256") != hashlib.sha256(input_path.read_bytes()).hexdigest():
            raise ValueError("existing partial output belongs to a different input manifest")
    completed = {row["comparison_id"]: row for row in (prior or {}).get("comparisons", [])}
    new_count = 0
    client = boto3.client("bedrock-runtime", region_name=REGION,
                          config=Config(retries={"max_attempts": 1}, read_timeout=840, connect_timeout=10))
    for spec in specs:
        if spec["comparison_id"] in completed:
            continue
        for attempt in range(4):
            # The configured V3.2 in-region quota is ample; retain a small pause.
            time.sleep(interval_seconds)
            try:
                response = client.converse(
                    modelId=MODEL_ID,
                    system=[{"text": SYSTEM_PROMPT}],
                    messages=[{"role": "user", "content": [{"text": spec["prompt"]}]}],
                    inferenceConfig={"temperature": TEMPERATURE, "topP": TOP_P, "maxTokens": MAX_TOKENS},
                )
                break
            except Exception as exc:
                code = getattr(exc, "response", {}).get("Error", {}).get("Code")
                if code != "ThrottlingException" or attempt == 3:
                    raise
        else:  # pragma: no cover - loop either breaks or raises
            raise RuntimeError("DeepSeek request retry loop ended unexpectedly")
        contents = response["output"]["message"].get("content", [])
        visible_text = "\n".join(block["text"] for block in contents if isinstance(block.get("text"), str))
        parsed = parse_answer(visible_text)
        usage = response.get("usage", {})
        completed[spec["comparison_id"]] = {
            **{key: value for key, value in spec.items() if key != "prompt"},
            "prompt": spec["prompt"],
            "prompt_sha256": hashlib.sha256(spec["prompt"].encode("utf-8")).hexdigest(),
            "raw_visible_response": visible_text,
            "parsed": parsed,
            "usage": usage,
            "stop_reason": response.get("stopReason"),
            "request_id": response.get("ResponseMetadata", {}).get("RequestId"),
        }
        partial = {
            "experiment": input_data["experiment"],
            "status": "partial",
            "input_manifest_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "model": {"id": MODEL_ID, "region": REGION},
            "method": {
                "task": "blinded pairwise preference for ordinary-language quality",
                "system_prompt": SYSTEM_PROMPT,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens_including_reasoning": MAX_TOKENS,
                "ordering": "deterministic SHA-256 assignment; the request contains no candidate/control labels",
                "provider_seed": None,
                "in_region_requests_per_minute_quota": 100,
                "minimum_wait_before_each_request_seconds": interval_seconds,
                "preflight_cost_ceiling_usd": cost_bound,
                "input_token_bound_method": "UTF-8 byte count for fixed ASCII prompts; one byte per token upper bound, plus configured output-token cap",
                "cost_cap_usd": budget_usd,
            },
            "comparisons": [completed[key] for key in (spec["comparison_id"] for spec in specs) if key in completed],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(partial, ensure_ascii=False, indent=2) + "\n")
        new_count += 1
        if max_new_comparisons is not None and new_count >= max_new_comparisons:
            break

    rows = [completed[spec["comparison_id"]] for spec in specs if spec["comparison_id"] in completed]
    input_tokens = sum(row.get("usage", {}).get("inputTokens", 0) for row in rows)
    output_tokens = sum(row.get("usage", {}).get("outputTokens", 0) for row in rows)
    # AWS Bedrock V3.2 standard pricing: $0.00062/1K input, $0.00185/1K output.
    cost = input_tokens / 1000 * INPUT_USD_PER_1K + output_tokens / 1000 * OUTPUT_USD_PER_1K
    result = {
        "experiment": input_data["experiment"],
        "status": "complete" if len(rows) == len(specs) else "partial",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "model": {"id": MODEL_ID, "region": REGION},
        "method": {
            "task": "blinded pairwise preference for ordinary-language quality",
            "system_prompt": SYSTEM_PROMPT,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens_including_reasoning": MAX_TOKENS,
            "ordering": "deterministic SHA-256 assignment; the request contains no candidate/control labels",
            "provider_seed": None,
            "in_region_requests_per_minute_quota": 100,
            "minimum_wait_before_each_request_seconds": interval_seconds,
            "preflight_cost_ceiling_usd": cost_bound,
            "input_token_bound_method": "UTF-8 byte count for fixed ASCII prompts; one byte per token upper bound, plus configured output-token cap",
            "cost_cap_usd": budget_usd,
            "pricing_source": "AWS Bedrock pricing page, retrieved 2026-09-25; V3.2 $0.62/M input and $1.85/M output",
        },
        "usage": {"comparisons": len(rows), "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens},
        "estimated_cost_usd": cost,
        "comparisons": rows,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_MIN_INTERVAL_SECONDS)
    parser.add_argument("--max-new-comparisons", type=int, default=None, help="limit this invocation; output remains resumable")
    args = parser.parse_args()
    report = run(args.input, args.output, args.budget_usd, args.interval_seconds, args.max_new_comparisons)
    print(json.dumps({"output": str(args.output), "status": report["status"], "comparisons": report["usage"]["comparisons"], "usage": report["usage"], "estimated_cost_usd": report["estimated_cost_usd"]}, indent=2))


if __name__ == "__main__":
    main()
