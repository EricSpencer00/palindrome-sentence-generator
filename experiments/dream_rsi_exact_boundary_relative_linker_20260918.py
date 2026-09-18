"""Third Dream-RSI exact-boundary repair: a relative-clause linker frame."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.dream_rsi_exact_boundary_20260918 import (  # noqa: E402
    FRESH,
    LEFT_FRAME,
    audit,
    discover,
)

EXPERIMENT = "dream-rsi-exact-boundary-relative-linker-20260918"
RIGHT_FRAME = ("det", "subject", "linker", "verb", "name")


def _controls() -> list[dict]:
    texts = (
        "The captain who marks maps; a writer reads Iris.",
        "A gardener that opens doors; the pilot guards Nora.",
        "Some clerks who chart plans; a baker finds Rhea.",
    )
    return [
        {
            "candidate_id": f"linker-control-{i}",
            "rendered": text,
            "audit": audit(text),
            "reader_status": "human-unreviewed",
            "provenance": {
                "fresh_authored_control": True,
                "heldout_relative_linker_frame": True,
                "catalogue_used": False,
                "finished_tape_reversal": False,
                "repeated_self_palindromic_unit": False,
            },
        }
        for i, text in enumerate(texts)
    ]


def run() -> dict:
    bank = dict(FRESH)
    bank["linker"] = ("that", "who")
    policies = ("alphabetic", "rare_first", "boundary_first")
    reports = {
        policy: discover(bank, policy, right_frame=RIGHT_FRAME)
        for policy in policies
    }
    closures = [row for report in reports.values() for row in report["closures"]]
    controls = _controls()
    return {
        "experiment": EXPERIMENT,
        "method": "Dream-RSI exact-boundary grammar zipper with held-out relative linker frame",
        "construction": {
            "left_frame": list(LEFT_FRAME),
            "heldout_right_frame": list(RIGHT_FRAME),
            "repair_operator": "add a who/that relative linker before the right finite verb",
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
            "new_geometry": "held-out relative-clause linker role over the exact-boundary residual zipper",
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
            "operator": "jointly vary linker attachment and right subject number while preserving the residual state",
            "reason": "the held-out linker frame changes syntax geometry but has not yet closed a fresh tape",
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
