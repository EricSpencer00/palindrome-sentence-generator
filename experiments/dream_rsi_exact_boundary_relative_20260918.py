"""Held-out Dream-RSI repair: auxiliary/relative right grammar frame.

This is the first concrete repair after the exact-boundary zipper's fresh
zero-closure result.  It keeps the same outside-in character transition but
changes the held-out right derivation, adding an auxiliary and a finite verb;
the lexical bank and policy frontier are not silently widened by a duplicate
Cartesian sweep.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.dream_rsi_exact_boundary_20260918 import (
    FRESH,
    LEFT_FRAME,
    audit,
    discover,
    letters,
)

EXPERIMENT = "dream-rsi-exact-boundary-relative-20260918"
RIGHT_FRAME_HELDOUT = ("det", "subject", "aux", "verb", "name")


def _controls() -> list[dict]:
    texts = (
        "The captain has charted fresh maps; a writer was reading notes.",
        "A gardener did mark small stones; the pilot has opened doors.",
        "Some clerks will guard new books; a baker was writing plans.",
    )
    return [
        {
            "candidate_id": f"relative-control-{i}",
            "rendered": text,
            "audit": audit(text),
            "reader_status": "human-unreviewed",
            "provenance": {
                "fresh_authored_control": True,
                "heldout_right_frame": True,
                "catalogue_used": False,
                "finished_tape_reversal": False,
                "repeated_self_palindromic_unit": False,
            },
        }
        for i, text in enumerate(texts)
    ]


def run() -> dict:
    bank = dict(FRESH)
    bank["aux"] = ("did", "has", "was", "will")
    policies = ("alphabetic", "rare_first", "boundary_first")
    reports = {
        policy: discover(bank, policy, right_frame=RIGHT_FRAME_HELDOUT)
        for policy in policies
    }
    closures = [row for report in reports.values() for row in report["closures"]]
    controls = _controls()
    return {
        "experiment": EXPERIMENT,
        "method": "Dream-RSI exact-boundary grammar zipper with held-out auxiliary frame",
        "construction": {
            "left_frame": list(LEFT_FRAME),
            "heldout_right_frame": list(RIGHT_FRAME_HELDOUT),
            "repair_operator": "add an auxiliary plus finite-verb role on the right syntax edge",
            "live_residual_equality": True,
            "mismatch_pruned_before_render": True,
        },
        "policy_replays": reports,
        "rendered_candidates": controls,
        "fresh_exact_closures": closures,
        "stats": {
            "fresh_nodes": sum(report["stats"]["nodes"] for report in reports.values()),
            "fresh_exact": len(closures),
            "longest_control_letters": max(row["audit"]["letters"] for row in controls),
            "policy_count": len(policies),
        },
        "novelty_preflight": {
            "new_geometry": "held-out auxiliary/finite-verb right grammar frame over the exact-boundary residual zipper",
            "prior_lane_reused": False,
            "duplicate_sweep": False,
            "catalogue_used": False,
        },
        "reader_gate": {
            "status": "not_triggered" if not closures else "human_blind_review_required",
            "programmatic_metrics_are_diagnostic": True,
            "reason": "no fresh exact closure" if not closures else "fresh exact closure requires independent mechanical and novelty gates",
        },
        "next_repair": {
            "operator": "replace one auxiliary with a relative-clause linker while preserving the live residual and role parse",
            "reason": "held-out auxiliary frame is a distinct construction transition, but no exact fresh closure has appeared",
        },
        "provenance": {
            "fresh_bank_authored_for_run": True,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "human_readability_certified": False,
        },
    }


if __name__ == "__main__":
    payload = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        directory.mkdir(exist_ok=True)
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], indent=2))
