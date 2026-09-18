"""Validate and summarize ratings for a frozen readability-length study."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.revision_agreement import ordinal_alpha


DIMENSIONS = ("grammaticality", "subject", "coherence")


def rating(value: str) -> int | None:
    if not value.strip():
        return None
    parsed = int(value)
    if parsed not in range(4):
        raise ValueError("ratings must be integers from 0 through 3")
    return parsed


def analyze(run_dir: Path) -> dict:
    raise RuntimeError(
        "legacy readability-length analysis is retired: its v3/catalogue packet is non-distributable"
    )
    key = {row["id"]: row for row in json.loads((run_dir / "key.json").read_text())}
    ratings: list[dict] = []
    with (run_dir / "human-ratings.csv").open(newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["item_id"] not in key:
                raise ValueError(f"unknown item ID: {raw['item_id']}")
            row = {"rater_id": raw["rater_id"].strip(), "item_id": raw["item_id"]}
            row.update({name: rating(raw[name]) for name in DIMENSIONS})
            if row["rater_id"] and any(row[name] is not None for name in DIMENSIONS):
                ratings.append(row)
    raters = sorted({row["rater_id"] for row in ratings})
    if len(raters) < 3:
        return {"status": "pending", "raters_with_scores": len(raters),
                "required_independent_raters": 3, "rated_rows": len(ratings)}

    seen: set[tuple[str, str]] = set()
    by_item: dict[str, list[dict]] = defaultdict(list)
    for row in ratings:
        marker = (row["rater_id"], row["item_id"])
        if marker in seen:
            raise ValueError(f"duplicate rating: {marker}")
        seen.add(marker)
        by_item[row["item_id"]].append(row)

    summaries = {}
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item_id, row in key.items():
        groups[row["source"], row["band"]].append(item_id)
    for group, item_ids in sorted(groups.items()):
        label = "/".join(group)
        rows = [rating_row for item_id in item_ids for rating_row in by_item[item_id]]
        summaries[label] = {
            "items": len(item_ids),
            "ratings": len(rows),
            **{dimension: (fmean([row[dimension] for row in rows
                                  if row[dimension] is not None])
                           if any(row[dimension] is not None for row in rows) else None)
               for dimension in DIMENSIONS},
        }

    reliability = {}
    for dimension in DIMENSIONS:
        matrix = []
        for item_id in key:
            per_rater = {row["rater_id"]: row[dimension] for row in by_item[item_id]}
            matrix.append([per_rater.get(rater_id) for rater_id in raters])
        reliability[dimension] = ordinal_alpha(matrix)
    return {"status": "complete", "raters_with_scores": len(raters),
            "rated_rows": len(ratings), "summary": summaries,
            "ordinal_krippendorff_alpha": reliability}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raise RuntimeError(
        "legacy readability-length analysis is retired: its v3/catalogue packet is non-distributable"
    )
    result = analyze(args.run_dir)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
