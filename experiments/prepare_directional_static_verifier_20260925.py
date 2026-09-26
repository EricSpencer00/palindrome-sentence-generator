"""Freeze GPT-2, Brown, and DeepSeek inputs for v4 selected outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.paired_directional_search_resource_sensitivity_20260925 import (  # noqa: E402
    sha256_file,
    verify_frozen_manifest,
)
from experiments.static_language_verifier_20260925 import (  # noqa: E402
    SHUFFLES_PER_ITEM,
    sha256_text,
    shuffle_surface,
    stable_shuffle_seed,
)

GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DEEPSEEK_MODEL_ID = "deepseek.v3.2"
DEEPSEEK_REGION = "us-east-2"
DEEPSEEK_BUDGET_USD = 0.20


def prepare(manifest_path: Path, result_path: Path, output_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    verify_frozen_manifest(manifest)
    result = json.loads(result_path.read_text())
    manifest_hash = sha256_file(manifest_path)
    if result.get("experiment") != manifest["experiment"]:
        raise ValueError("search result experiment does not match the v4 manifest")
    if result.get("phase") != "confirmatory":
        raise ValueError("static verifier inputs must be built from confirmatory outputs")
    if result.get("protocol_manifest_sha256") != manifest_hash:
        raise ValueError("search result was not generated from this frozen manifest")

    expected = {
        (job["pair_id"], job["resource_configuration"], job["method"])
        for job in manifest["jobs"] if job["phase"] == "confirmatory"
    }
    observed = {
        (row["pair_id"], row["resource_configuration"], row["method"])
        for row in result["results"]
    }
    if expected != observed:
        raise ValueError("search result rows do not match confirmatory jobs")

    candidates = [row for row in result["results"] if row.get("text")]
    if not candidates:
        raise ValueError("no selected outputs are available for static verification")
    items = []
    for row in candidates:
        if not row["independent_exactness_check"] or not all(row["basic_structural_checks"].values()):
            raise ValueError(f"refusing invalid selected output: {row['pair_id']}")
        task_id = ":".join((
            f"seed-{row['seed']}", f"band-{row['band_id']}",
            row["resource_configuration"], row["method"],
        ))
        candidate_id = f"candidate:{task_id}"
        candidate = {
            "id": candidate_id,
            "kind": "candidate",
            "pair_id": task_id,
            "text": row["text"],
            "word_count": row["word_count"],
            "text_sha256": sha256_text(row["text"]),
            "source": {
                "search_experiment": result["experiment"],
                "search_manifest_sha256": manifest_hash,
                "search_result_sha256": sha256_file(result_path),
                "seed": row["seed"],
                "band_id": row["band_id"],
                "resource_configuration": row["resource_configuration"],
                "method": row["method"],
                "text_sha256": row["text_sha256"],
            },
        }
        items.append(candidate)
        for replicate in range(SHUFFLES_PER_ITEM):
            shuffle_id = f"shuffle:candidate:{task_id}:{replicate:02d}"
            seed = stable_shuffle_seed(task_id, replicate)
            shuffled = shuffle_surface(row["text"], seed)
            items.append({
                "id": shuffle_id,
                "kind": "candidate_same_word_shuffle",
                "pair_id": task_id,
                "text": shuffled,
                "word_count": row["word_count"],
                "text_sha256": sha256_text(shuffled),
                "source": {
                    "candidate_id": candidate_id,
                    "replicate": replicate,
                    "seed": seed,
                    "preserves": "word multiset, punctuation, whitespace, and each word's original casing",
                },
            })

    output = {
        "experiment": "directional-search-static-verifier-20260925",
        "status": "frozen_confirmatory_outputs_and_deterministic_controls",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": manifest_hash,
        "search_result_sha256": sha256_file(result_path),
        "candidate_count": len(candidates),
        "shuffle_count_per_candidate": SHUFFLES_PER_ITEM,
        "sources": {
            "search_manifest": manifest_path.as_posix(),
            "search_result": result_path.as_posix(),
            "input_builder_sha256": sha256_file(Path(__file__)),
            "shuffle_helper_sha256": sha256_file(ROOT / "experiments/static_language_verifier_20260925.py"),
            "brown_diagnostic_sha256": sha256_file(ROOT / "experiments/score_length_stratified_readability.py"),
            "gpt2_scorer_sha256": sha256_file(ROOT / "experiments/score_gpt2_static_verifier_20260925.py"),
            "deepseek_runner_sha256": sha256_file(ROOT / "experiments/infer_deepseek_static_verifier_20260925.py"),
        },
        "method": {
            "brown": "Brown training-document word-bigram model; score each candidate against 32 deterministic same-word shuffles using the pinned sentence-aware diagnostic",
            "gpt2": {
                "model": "openai-community/gpt2",
                "revision": GPT2_REVISION,
                "metric": "mean causal-token log probability; higher is better model fit, not readability",
                "items_scored": "each selected output and all 32 same-word shuffles",
                "device": "NUC bench CPU",
            },
            "deepseek": {
                "model_id": DEEPSEEK_MODEL_ID,
                "region": DEEPSEEK_REGION,
                "task": "one blinded candidate-versus-replicate-00 same-word-shuffle comparison per selected output",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_output_tokens": 256,
                "provider_seed": None,
                "max_spend_usd": DEEPSEEK_BUDGET_USD,
                "limitation": "one model's prompted preference; not human ratings or a validated readability measure",
            },
            "shuffle_seed": 20260925,
            "shuffle_seed_derivation": "first 8 SHA-256 bytes of '<base>:<task-id>:<replicate>', big-endian",
            "repeatability": "candidate text, generated shuffles, model revision, prompt, and inference parameters are frozen; Bedrock does not expose a seed that guarantees the same response on rerun",
        },
        "items": items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.manifest, args.results, args.output)
    print(json.dumps({
        "output": str(args.output),
        "candidates": result["candidate_count"],
        "passages_with_shuffles": len(result["items"]),
        "deepseek_bound_usd": result["method"]["deepseek"]["max_spend_usd"],
    }, indent=2))


if __name__ == "__main__":
    main()
