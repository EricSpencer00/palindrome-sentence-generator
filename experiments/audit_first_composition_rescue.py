"""Record programmatic diagnostics for a frozen two-pair material screen.

This deliberately has no readability conclusion: a material-and-order screen
contains no prose/shuffle positive controls.  The values are recorded to decide
whether costly guidance or human screening is worthwhile, not to select a
winner.  It accepts either a saved display (``rendered``) or the frozen plain
word run, which lets it diagnose a boundary experiment without asking a second
presenter to decide its punctuation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.audit_programmatic_readability import BrownBigramModel, FEATURES, item_features


def audit(run_dir: Path, seed: int = 20260912, shuffles: int = 32) -> dict:
    materials = json.loads((run_dir / "internal" / "materials.json").read_text())
    model = BrownBigramModel.from_brown()
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in materials["variants"]:
        source_arm = item.get("source_arm", "rescue")
        group = f"{source_arm}:{item['condition']}"
        row = {**item, "id": item.get(
            "id", f"{source_arm}:{item['block_id']}:{item['condition']}"),
            "source": source_arm, "band": item["condition"]}
        groups[group].append(item_features(
            row, item.get("rendered", item["plain"]), model, seed, shuffles))
    return {
        "status": "diagnostic_only_no_human_or_control_readability_claim",
        "method": {
            "brown_bigram": "add-alpha word bigram model trained on NLTK Brown",
            "order_gain": "observed mean log probability minus mean over own-word shuffles",
            "shuffle_count": shuffles,
            "random_seed": seed,
        },
        "limits": [
            "This screen does not include intact prose/shuffle controls, so its metrics are not calibrated as a readability evaluator.",
            "A low score is a local-order warning only; a high score would not establish grammaticality, subject, or coherence.",
        ],
        "group_means": {
            condition: {"n": len(rows), **{
                feature: statistics.fmean(row[feature] for row in rows)
                for feature in FEATURES
            }}
            for condition, rows in sorted(groups.items())
        },
        "items": [row for rows in groups.values() for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--shuffles", type=int, default=32)
    args = parser.parse_args()
    if args.shuffles < 2:
        parser.error("--shuffles must be at least 2")
    report = audit(args.run_dir, args.seed, args.shuffles)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "groups": len(report["group_means"])}, indent=2))


if __name__ == "__main__":
    main()
