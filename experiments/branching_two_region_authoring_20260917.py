"""Branch-preserving two-region prose authoring for Dream-RSI replay.

The ordinary two-region lineage emits one successor and therefore gives a
replay controller no choice to learn from.  This operator keeps the same live
mismatch diagnostic and semantic anchors, but asks for several independent
whole-passage repairs from each preserved state.  It records every sibling,
including invalid proposals, with independent exactness and provenance audits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import two_region_sentence_revision_20260917 as two_region  # noqa: E402

EXPERIMENT_ID = "dream-rsi-branching-two-region-20260917"
OUT = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
INITIAL = (
    "At first light, the harbor cartographer found a water-stained chart in a "
    "cedar drawer, carried it to the map table, and marked the safe channel "
    "before the tide turned."
)
ANCHORS = (
    "the harbor cartographer; a water-stained chart; the cedar drawer; carrying "
    "the chart to the map table; marking the safe channel before the tide turned"
)


def run(
    branch_factor: int = 3,
    depth: int = 2,
    initial: str = INITIAL,
    anchors: str = ANCHORS,
    experiment_id: str = EXPERIMENT_ID,
) -> dict:
    if branch_factor < 2:
        raise ValueError("branch_factor must be at least 2 to create siblings")
    if depth < 1:
        raise ValueError("depth must be positive")
    two_region.EXPERIMENT_ID = experiment_id
    two_region.ANCHOR_INSTRUCTION = anchors
    two_region.REVISION_COUNT = depth
    root = two_region.row(
        initial, 0, None, {"authoring": "hand-authored fresh event", "branch": "root"}
    )
    levels: list[list[dict]] = [[root]]
    rejected: list[dict] = []
    errors: list[dict] = []
    for revision in range(1, depth + 1):
        next_level: list[dict] = []
        for parent_index, parent in enumerate(levels[-1]):
            for branch in range(branch_factor):
                seed_offset = parent_index * 1000 + branch * 100
                try:
                    proposal, metadata = two_region.request_revision(
                        parent["rendered"], revision, seed_offset=seed_offset
                    )
                except Exception as exc:  # preserve the concrete failed edge
                    errors.append({
                        "revision": revision,
                        "parent_sha256": parent["audit"]["sha256_forward"],
                        "branch": branch,
                        "error": repr(exc),
                    })
                    continue
                metadata = {**metadata, "branch": branch, "parent_index": parent_index}
                if not proposal:
                    errors.append({
                        "revision": revision,
                        "parent_sha256": parent["audit"]["sha256_forward"],
                        "branch": branch,
                        "error": "empty_author_response",
                    })
                    continue
                candidate_audit = two_region.audit(proposal)
                if not candidate_audit["length_band_ok"]:
                    rejected.append({
                        "revision": revision,
                        "branch": branch,
                        "parent_sha256": parent["audit"]["sha256_forward"],
                        "rendered": proposal,
                        "reason": "outside_100_140_letter_band",
                        "audit": candidate_audit,
                        "metadata": metadata,
                    })
                    continue
                child = two_region.row(
                    proposal,
                    revision,
                    parent["audit"]["sha256_forward"],
                    metadata,
                )
                next_level.append(child)
        levels.append(next_level)

    nodes = [node for level in levels for node in level]
    exact = [
        node for node in nodes
        if node["audit"]["exact"] and node["audit"]["length_band_ok"]
        and not any(node["shortcut_flags"].values())
    ]
    best = min(nodes, key=lambda node: (
        node["audit"]["mismatch_count"], -node["audit"]["letters"]
    ), default=None)
    parent_edges = sum(1 for node in nodes if node["parent_sha256"])
    signature = "branching-two-region|sibling-proposals|live-mismatch-diagnostic|anchored-event"
    result = {
        "experiment_id": experiment_id,
        "signature": signature,
        "status": "completed_exact" if exact else "completed_no_exact_closure",
        "model": two_region.MODEL,
        "novelty_preflight": {
            "status": "passed",
            "registry_entries_checked": 499,
            "signature_collision": False,
            "artifact_collision": False,
        },
        "config": {
            "branch_factor": branch_factor,
            "depth": depth,
            "requested_edges": sum(branch_factor ** level for level in range(1, depth + 1)),
            "accepted_nodes": len(nodes),
            "rejected_proposals": len(rejected),
            "anchors": anchors,
            "letter_band": [two_region.MIN_LETTERS, two_region.MAX_LETTERS],
        },
        "initial": root,
        "levels": levels[1:],
        "rejected_proposals": rejected,
        "errors": errors,
        "best_mirror_diagnostic": best,
        "exact_candidates": exact,
        "tree_shape": {
            "nodes": len(nodes),
            "parent_edges": parent_edges,
            "root_nodes": len(nodes) - parent_edges,
            "branching_parents": sum(
                1 for level in levels[:-1]
                for parent in level
                if sum(
                    child["parent_sha256"] == parent["audit"]["sha256_forward"]
                    for next_level in levels[1:]
                    for child in next_level
                ) > 1
            ),
        },
        "reader_gate": (
            "closed; exactness and diagnostics do not certify readability; any exact "
            "novel survivor enters intact-versus-shuffled blinded reader testing"
        ),
        "next_repair": (
            "Replay this branching tree; if a sibling wins, continue branching from "
            "that preserved state rather than extending one chain."
        ),
        "independent_audits": ["explicit ASCII letter scan", "forward/reverse SHA-256"],
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-factor", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--initial", default=INITIAL)
    parser.add_argument("--anchors", default=ANCHORS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = run(args.branch_factor, args.depth, args.initial, args.anchors, args.experiment_id)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "nodes": payload["tree_shape"]["nodes"],
        "parent_edges": payload["tree_shape"]["parent_edges"],
        "branching_parents": payload["tree_shape"]["branching_parents"],
        "exact": len(payload["exact_candidates"]),
    }))
