"""Fail-closed preflight for a proposed construction experiment.

The registry is intentionally checked before a generator is run.  A new
filename, seed, beam width, or larger lexical bank is not enough: the
candidate signature and artifact path must be absent from both the retained
families and the explicit exclusion ledger.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"

# These words describe the shared palindrome bookkeeping rather than a route's
# construction dimension.  They match the audit tool's vocabulary so a
# preflight and the retained report cannot disagree about why two proposals
# look similar.
COMMON_ATOMS = {
    "a", "an", "and", "after", "audit", "authoring", "before", "character",
    "complete", "construction", "constraints", "cross", "derived", "derivation",
    "english", "equation", "exact", "final", "from", "fresh", "full", "generation",
    "global", "grammar", "held", "heldout", "in", "independent", "join", "joint",
    "left", "lexical", "lexicalized", "of", "order", "out", "over", "paired", "parse",
    "parser", "repair", "residual", "reverse", "right", "sentence", "state", "surface",
    "tape", "the", "through", "to", "typed", "unit", "word", "words", "with",
}


def signature_atoms(signature: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", signature.lower())
        if token and token not in COMMON_ATOMS
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def preflight(
    experiment_id: str,
    signature: str,
    artifact: str,
    *,
    near_threshold: float = 0.40,
) -> dict:
    data = json.loads(REGISTRY.read_text())
    retained = data.get("entries", [])
    excluded = data.get("excluded", [])
    all_rows = [("registered", row) for row in retained]
    all_rows.extend(("excluded", row) for row in excluded)

    collisions = []
    for kind, row in all_rows:
        fields = []
        if row.get("id") == experiment_id:
            fields.append("id")
        if row.get("signature") == signature:
            fields.append("signature")
        if row.get("artifact") == artifact:
            fields.append("artifact")
        if fields:
            collisions.append({"kind": kind, "id": row.get("id"), "fields": fields})
    if collisions:
        raise ValueError(f"novelty collision: {collisions}")

    artifact_path = ROOT / artifact
    if artifact_path.exists():
        raise ValueError(f"artifact already exists; choose a new path: {artifact}")

    # Exact collisions are hard failures above.  This second screen is
    # deliberately advisory: a genuinely different state-space dimension can
    # share words such as "semantic" or "clause", but the proposer must see
    # the nearest prior family before running and record the disposition in
    # the run artifact.  Keeping this in preflight closes the old loophole in
    # which only the post-hoc audit knew that a proposal was close to a prior
    # route.
    atoms = signature_atoms(signature)
    near = []
    for kind, row in all_rows:
        score = _jaccard(atoms, signature_atoms(row.get("signature", "")))
        if score >= near_threshold:
            near.append({
                "kind": kind,
                "id": row.get("id"),
                "jaccard": round(score, 6),
                "shared_atoms": sorted(atoms & signature_atoms(row.get("signature", ""))),
            })
    near.sort(key=lambda row: (-row["jaccard"], row["id"] or ""))
    return {
        "status": "novel",
        "experiment_id": experiment_id,
        "signature": signature,
        "artifact": artifact,
        "registered_families_checked": len(retained),
        "excluded_routes_checked": len(excluded),
        "conceptual_near_pairs": near,
        "manual_review_required": bool(near),
        "near_threshold": near_threshold,
        "review_policy": (
            "A near pair is not an automatic rejection, but the proposer must "
            "document a different construction dimension or classify the run "
            "as a repair before execution."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, dest="experiment_id")
    parser.add_argument("--signature", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--near-threshold", type=float, default=0.40)
    args = parser.parse_args()
    print(json.dumps(preflight(
        args.experiment_id, args.signature, args.artifact,
        near_threshold=args.near_threshold,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
