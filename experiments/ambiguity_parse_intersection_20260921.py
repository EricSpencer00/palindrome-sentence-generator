"""Preflight the proposed garden-path parse-intersection construction.

The proposed operator is already present in the retained homograph-sense
lattice: one orthographic tape, two independent semantic parses, and
sense-conditioned word-boundary choice.  Therefore this file records a
duplicate decision and deliberately emits no candidates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
OUT = ROOT / "runs" / "ambiguity-parse-intersection-20260921.json"
TARGETS = [
    "homograph-sense-lattice-20260915",
    "denotational-equality-registry-new-20260920",
]


def main() -> None:
    registry = json.loads(REGISTRY.read_text())
    all_entries = registry["entries"] + registry.get("excluded", [])
    found = {entry["id"]: entry for entry in all_entries}
    overlaps = [found[key] for key in TARGETS if key in found]
    payload = "\n".join(
        f"{item['id']}|{item.get('signature', '')}|{item.get('distinction', '')}"
        for item in overlaps
    ).encode()
    result = {
        "experiment_id": "ambiguity-parse-intersection-20260921",
        "method": "garden-path/syntactic-ambiguity with shared orthographic tape",
        "status": "duplicate_preflight_only",
        "novelty_preflight": {
            "proposed_operator": [
                "one orthographic tape",
                "two ordinary semantic/syntactic parses",
                "ambiguity-carried roles rather than mirrored units",
            ],
            "overlap_ids": [item["id"] for item in overlaps],
            "overlap_sha256": hashlib.sha256(payload).hexdigest(),
            "fresh_operator": False,
            "reason": (
                "homograph-sense-lattice already specifies one orthographic tape "
                "with two independent semantic parses and sense-conditioned word "
                "boundaries; denotational equality separately covers alternate "
                "replayable syntax topologies. A new garden-path run would overlap."
            ),
        },
        "overlap_records": [
            {
                "id": item["id"],
                "signature": item.get("signature"),
                "artifact": item.get("artifact"),
                "run_artifacts": item.get("run_artifacts", []),
                "status": item.get("status"),
            }
            for item in overlaps
        ],
        "rendered_controls": [],
        "exact_candidates": [],
        "independent_two_pointer_sha_audit": {
            "performed": False,
            "reason": "No candidate was generated after duplicate preflight.",
        },
        "provenance": {
            "source_text": "none",
            "catalogue_text": False,
            "borrowed_text": False,
            "post_hoc_repair": False,
            "rlaif_per_candidate": False,
        },
        "next_construction": (
            "Reject this lane; preflight a surface-generation operator that is "
            "not orthographic ambiguity, CFG ambiguity, homograph sense choice, "
            "or denotational topology selection."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "overlaps": len(overlaps)}))


if __name__ == "__main__":
    main()
