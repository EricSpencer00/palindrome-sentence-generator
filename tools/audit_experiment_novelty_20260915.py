"""Audit construction-family novelty beyond exact registry collisions.

The registry's preflight rejects exact id/signature/artifact reuse.  This
report adds a deterministic lexical-overlap screen so a proposed route cannot
look new merely because it renames a seed, beam, or lexical bank.  The overlap
screen is advisory: construction families still require a human decision
about whether the state-space dimension is genuinely different.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"

# Terms that describe the shared palindrome bookkeeping rather than a route's
# construction dimension.  Keeping these out makes the report useful without
# pretending that lexical token overlap alone proves equivalence.
COMMON = {
    "a", "an", "and", "after", "audit", "authoring", "before", "character",
    "complete", "construction", "constraints", "cross", "derived", "derivation",
    "english", "equation", "exact", "final", "from", "fresh", "full", "generation",
    "global", "grammar", "held", "heldout", "in", "independent", "join", "joint",
    "left", "lexical", "lexicalized", "of", "order", "out", "over", "paired", "parse",
    "parser", "repair", "residual", "reverse", "right", "sentence", "state", "surface",
    "tape", "the", "through", "to", "typed", "unit", "word", "words", "with",
}


def atoms(signature: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", signature.lower())
        if token and token not in COMMON
    }


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def audit(path: Path = REGISTRY, threshold: float = 0.40) -> dict:
    data = json.loads(path.read_text())
    entries = data.get("entries", [])
    excluded = data.get("excluded", [])
    rows = []
    near_pairs = []
    for index, row in enumerate(entries):
        current = atoms(row["signature"])
        prior = []
        for prior_row in entries[:index]:
            score = jaccard(current, atoms(prior_row["signature"]))
            prior.append({
                "id": prior_row["id"],
                "jaccard": round(score, 6),
                "shared_atoms": sorted(current & atoms(prior_row["signature"])),
            })
            if score >= threshold:
                near_pairs.append({
                    "newer": row["id"],
                    "older": prior_row["id"],
                    "jaccard": round(score, 6),
                    "shared_atoms": sorted(current & atoms(prior_row["signature"])),
                })
        nearest = max(prior, key=lambda value: value["jaccard"], default=None)
        rows.append({
            "id": row["id"],
            "signature_atoms": sorted(current),
            "nearest_prior": nearest,
            "manual_review_required": bool(nearest and nearest["jaccard"] >= threshold),
        })
    return {
        "registry": str(path.relative_to(ROOT)),
        "registered_entries": len(entries),
        "excluded_routes": len(excluded),
        "threshold": threshold,
        "exact_signature_collisions": len({r["signature"] for r in entries}) != len(entries),
        "near_pairs": near_pairs,
        "entries": rows,
        "policy": (
            "Exact collisions fail closed. A near pair is a human-review flag, "
            "not evidence of novelty; the proposer must document a different "
            "construction state-space dimension or classify the run as a repair."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.40)
    args = parser.parse_args()
    result = audit(threshold=args.threshold)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
