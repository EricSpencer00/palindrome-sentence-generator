#!/usr/bin/env python3
"""Recompute saved Bedrock Qwen3 readability-audit summaries offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from local_llm_readability_audit_20260926 import compute_analysis  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(path: Path, verify_sources: bool = False) -> dict[str, Any]:
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("status") != "complete":
        raise ValueError("saved run is not complete")
    presentations = run.get("presentations")
    responses = run.get("responses")
    if not isinstance(presentations, list) or not isinstance(responses, list):
        raise ValueError("run lacks the frozen presentation manifest or saved responses")

    presentation_ids = [row.get("presentation_id") for row in presentations]
    response_ids = [row.get("presentation_id") for row in responses]
    if len(presentation_ids) != 244 or len(set(presentation_ids)) != 244:
        raise ValueError("expected 244 unique presentations")
    if len(response_ids) != 244 or len(set(response_ids)) != 244:
        raise ValueError("expected one successful response per presentation")
    if set(presentation_ids) != set(response_ids):
        raise ValueError("response IDs do not match the frozen presentation manifest")
    presentation_by_id = {row["presentation_id"]: row for row in presentations}
    for response in responses:
        presentation = presentation_by_id[response["presentation_id"]]
        if response.get("text_sha256") != presentation.get("text_sha256"):
            raise ValueError("response text hash does not match its presentation")
        scores = response.get("scores")
        if not isinstance(scores, dict):
            raise ValueError("response has no parsed rubric scores")
        for dimension in ("grammaticality", "coherent_meaning"):
            score = scores.get(dimension)
            if type(score) is not int or not 0 <= score <= 3:
                raise ValueError(f"invalid saved {dimension} score")

    analysis = compute_analysis(presentations, responses)
    if analysis != run.get("analysis"):
        raise ValueError("recomputed summaries differ from the saved run analysis")
    if analysis["candidate_unique_texts"]["n"] != 52:
        raise ValueError("expected 52 unique candidate texts")
    if analysis["candidate_matched_shuffles_3_per_text"]["n"] != 156:
        raise ValueError("expected three matched shuffles for each candidate")

    source_checks: dict[str, dict[str, str | bool]] = {}
    if verify_sources:
        artifact_map = {
            "candidate_packet": "candidate_packet_sha256",
            "static_verifier_input": "static_verifier_input_sha256",
            "heldout_brown_control_manifest": "heldout_brown_control_manifest_sha256",
        }
        artifacts = run.get("source_artifacts", {})
        for label, key in artifact_map.items():
            recorded = artifacts.get(key)
            relative = artifacts.get(label)
            if not isinstance(recorded, str) or not isinstance(relative, str):
                raise ValueError(f"missing source-artifact provenance for {label}")
            source = ROOT / relative
            observed = sha256_file(source)
            source_checks[label] = {
                "recorded_sha256": recorded,
                "observed_sha256": observed,
                "match": recorded == observed,
            }
        if not all(row["match"] for row in source_checks.values()):
            raise ValueError("source-file hashes differ from the historical run")

    candidate = analysis["candidate_unique_texts"]
    shuffles = analysis["candidate_matched_shuffles_3_per_text"]
    attempts = len(responses) + len(run.get("failed_attempts", []))
    max_request_cost = run.get("pricing", {}).get("per_request_worst_case_usd")
    expected_reserve = round(attempts * max_request_cost, 9) if max_request_cost else None
    if expected_reserve is not None and abs(run.get("budget_reserved_usd", 0) - expected_reserve) > 1e-9:
        raise ValueError("cumulative worst-case reservation does not match recorded attempts")

    return {
        "experiment_id": run.get("experiment_id"),
        "status": run["status"],
        "model_id": run.get("model_facts", {}).get("model_id"),
        "region": run.get("model_facts", {}).get("region"),
        "presentations": len(presentations),
        "successful_responses": len(responses),
        "total_attempts_including_interrupted": attempts,
        "candidate_texts": candidate,
        "matched_shuffles": shuffles,
        "calibration": run.get("calibration"),
        "repeat_agreement": analysis["hidden_repeat_agreement"],
        "analysis_recomputed_exactly": True,
        "source_file_checks": source_checks or None,
        "known_token_cost_subtotal_usd": run.get("known_token_cost_subtotal_usd"),
        "actual_cost_estimate_complete": run.get("actual_cost_estimate_complete"),
        "worst_case_budget_reserved_usd": run.get("budget_reserved_usd"),
        "worst_case_reservation_matches_attempt_count": expected_reserve is not None,
        "run_json_sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--verify-sources",
        action="store_true",
        help="reconstruct local source hashes; requires the exact historical inputs",
    )
    args = parser.parse_args()
    print(json.dumps(summarize(args.input, args.verify_sources), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
