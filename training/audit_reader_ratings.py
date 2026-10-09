"""Join scalar reader ratings to passages and separate exactness from readability.

Run: python -m training.audit_reader_ratings RATINGS --packet PACKET --out OUT
Packet format: {"study": "...", "items": [{"item": "...", "text": "..."}]}.
Missing passages never enter the exact-only dataset. Non-palindromes remain in
an audit file, because they may be deliberate prose or shuffled controls.
These scalar ratings are not A/B choices and are not converted into DPO labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics

from llm_palindrome.validator import is_palindrome, normalize


def summarize(rows):
    return {
        "count": len(rows),
        "mean_grammar": statistics.mean(r["grammar"] for r in rows) if rows else None,
        "mean_meaning": statistics.mean(r["meaning"] for r in rows) if rows else None,
        "both_3": sum(r["grammar"] == r["meaning"] == 3 for r in rows),
        "meaning_1": sum(r["meaning"] == 1 for r in rows),
    }


def audit(ratings, packet=None):
    study = ratings.get("study")
    if not isinstance(study, str) or not isinstance(ratings.get("rows"), list):
        raise ValueError("ratings must contain study and rows")
    items = {}
    if packet is not None:
        if packet.get("study") != study:
            raise ValueError("packet study does not match ratings")
        for item in packet["items"]:
            key, text = item["item"], item["text"]
            if not isinstance(key, str) or not isinstance(text, str) or not text.strip():
                raise ValueError("each packet item needs an ID and nonempty text")
            if key in items:
                raise ValueError(f"duplicate packet item: {key}")
            items[key] = text
    buckets = {"exact": [], "non_palindromes": [], "unresolved": []}
    seen = set()
    for row in ratings["rows"]:
        if not isinstance(row.get("rater"), str) or not isinstance(row.get("item"), str):
            raise ValueError("each rating needs rater and item IDs")
        key = row["rater"], row["item"]
        if key in seen:
            raise ValueError(f"duplicate rater/item: {key}")
        seen.add(key)
        for field in ("grammar", "meaning"):
            if type(row.get(field)) is not int or not 0 <= row[field] <= 3:
                raise ValueError(f"{field} must be an integer from 0 to 3")
        text = items.get(row["item"])
        if text is None:
            buckets["unresolved"].append(dict(row))
            continue
        tape = normalize(text)
        exact = is_palindrome(text)
        enriched = dict(row, text=text, letters=len(tape),
                        words=len(re.findall(r"[A-Za-z]+", text)), palindrome=exact)
        if not exact:
            mismatch = next((i for i in range(len(tape)) if tape[i] != tape[-1-i]), None)
            enriched["first_mismatch"] = (None if mismatch is None else {
                "left_index": mismatch, "right_index": len(tape)-1-mismatch,
                "left": tape[mismatch], "right": tape[-1-mismatch],
            })
        buckets["exact" if exact else "non_palindromes"].append(enriched)
    exact_rows = buckets["exact"]
    literal_association = [r for r in exact_rows if re.search(r"\bassociation\b", r["text"], re.I)]
    summary = {
        "study": study, "raters": len({r["rater"] for r in ratings["rows"]}),
        "all_ratings": summarize(ratings["rows"]),
        "counts": {k: len(v) for k, v in buckets.items()},
        "exact_only": summarize(exact_rows),
        "exact_length_bands": {
            label: summarize([r for r in exact_rows if low <= r["letters"] <= high])
            for label, low, high in [("1-49", 1, 49), ("50-119", 50, 119),
                                     ("120-299", 120, 299), ("300+", 300, float("inf"))]
        },
        "exact_literal_association": summarize(literal_association),
        "rating_pairs": dict(sorted(Counter(
            f'{r["grammar"]},{r["meaning"]}' for r in ratings["rows"]).items())),
        "notes": [
            "Meaning 1 means word associations with unclear relations; it is not a word count.",
            "Non-palindromes may be controls; excluded from exact-only data, retained for audit.",
            "Missing text is unresolved, never presumed exact.",
            "Ratings are development feedback; no DPO pairs or model weights are inferred.",
        ],
    }
    return dict(buckets, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ratings", type=Path)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.ratings.read_text()),
                   json.loads(args.packet.read_text()) if args.packet else None)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in result.items():
        (args.out / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(result["summary"]["counts"]))


if __name__ == "__main__":
    main()
