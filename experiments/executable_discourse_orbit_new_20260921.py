"""Novelty preflight for the proposed executable-discourse shared-orbit lane.

This file intentionally performs no duplicate generation.  The proposed
construction was checked against the retained experiment registry and is
already covered by three executable event/orbit lanes.  Keeping this as an
executable preflight makes the negative decision reproducible and prevents a
larger sweep from being misreported as progress.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
OUT = ROOT / "runs" / "executable-discourse-orbit-new-20260921.json"

PROPOSED_SIGNATURE = (
    "finite-model-precondition|joint-character-orbit|packed-event-grammar|"
    "entity-binding"
)
PROPOSED_FEATURES = {
    "finite-model-precondition",
    "joint-character-orbit",
    "packed-event-grammar",
    "entity-binding",
    "recipient-theme-state",
    "typed-pronoun-accessibility",
}
KNOWN_IDS = [
    "executable-discourse-shared-orbit-20260920",
    "executable-discourse-recipient-theme-20260920",
    "executable-discourse-pronoun-accessibility-20260920",
]


def main() -> None:
    registry = json.loads(REGISTRY.read_text())
    # The registry keeps historical retained-but-excluded experiments in a
    # second list; those are still authoritative for duplicate preflight.
    entries = registry["entries"] + registry.get("excluded", [])
    by_id = {entry["id"]: entry for entry in entries}
    overlaps = [by_id[item] for item in KNOWN_IDS if item in by_id]
    normalized = "\n".join(
        f"{entry['id']}|{entry.get('signature', '')}|{entry.get('distinction', '')}"
        for entry in overlaps
    ).encode()
    digest = hashlib.sha256(normalized).hexdigest()

    # Independent text audit is deliberately empty: this is a preflight, not
    # a candidate-producing run.  The referenced runs contain the actual
    # rendered controls and two-pointer/SHA audits.
    result = {
        "experiment_id": "executable-discourse-orbit-new-20260921",
        "method": "executable discourse constraints over shared character-orbit variables",
        "status": "duplicate_preflight_only",
        "proposed_signature": PROPOSED_SIGNATURE,
        "proposed_features": sorted(PROPOSED_FEATURES),
        "registry_sha256_of_overlap_records": digest,
        "overlap_records": [
            {
                "id": entry["id"],
                "signature": entry.get("signature"),
                "artifact": entry.get("artifact"),
                "run_artifacts": entry.get("run_artifacts", []),
                "status": entry.get("status"),
                "distinction": entry.get("distinction"),
            }
            for entry in overlaps
        ],
        "novelty_decision": {
            "fresh_operator": False,
            "reason": (
                "The proposed world-state/entity-binding/shared-orbit operator is "
                "already retained; recipient/theme and pronoun accessibility are "
                "also recorded successors. A new run would be a larger duplicate sweep."
            ),
            "excluded_from_method_count": True,
        },
        "rendered_controls": [],
        "exact_candidates": [],
        "independent_two_pointer_sha_audit": {
            "performed_in_this_preflight": False,
            "reason": "No text was generated because novelty preflight rejected the lane.",
        },
        "provenance": {
            "source_text": "none",
            "catalogue_text": False,
            "borrowed_text": False,
            "post_hoc_repair": False,
            "rlaif_per_candidate": False,
        },
        "next_construction": (
            "Do not widen this orbit family. Select a construction with a new "
            "surface-generation operator, then preflight it before implementation."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "overlaps": len(overlaps), "sha256": digest}))


if __name__ == "__main__":
    main()
