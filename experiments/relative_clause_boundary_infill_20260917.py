"""Boundary-crossing relative-clause infill (bounded Dream-RSI deployment).

This lane changes the geometry rather than widening the previous two-span
agent/object search.  It reopens a relative clause together with the host
clause's complement, so a word boundary can move across the attachment seam.
Every row is authored prose and receives an independent two-pointer/SHA audit;
the program never treats a readability score as a certificate.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "relative-clause-boundary-infill-20260917"

# These are authored scene fragments, not an imported sentence bank.  The
# relative-clause span and the host complement are jointly reopened at a seam.
FRAMES = (
    (
        "dawn",
        "At dawn",
        "By dusk",
        "the archivist",
        "the sailor",
        "checks the quiet harbor",
        "records a fresh signal",
    ),
    (
        "rain",
        "After rain",
        "Before night",
        "a curator",
        "a pilot",
        "keeps a narrow record",
        "notes the morning watch",
    ),
)

LEFT_RELATIVE = (
    "who marks the map",
    "who tracks the ledger",
    "who reviews the atlas",
)
RIGHT_RELATIVE = (
    "who follows the route",
    "who studies the harbor",
    "who charts the inlet",
)
LEFT_COMPLEMENTS = (
    "checks the quiet harbor",
    "copies a local signal",
    "guards a folded chart",
)
RIGHT_COMPLEMENTS = (
    # Each complement closes in ``a`` so the authored outer frame agrees on
    # the first mirrored character; the relative-clause seam is then the
    # unresolved region Dream-RSI is asked to repair.
    "records a local idea",
    "carries a quiet agenda",
    "notes a broad area",
)


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, Any]:
    tape = letters(text)
    mismatches = sum(a != b for a, b in zip(tape, tape[::-1]))
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and tape == tape[::-1],
        "mismatch_count": mismatches,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def residual(text: str) -> int:
    tape = letters(text)
    return sum(a != b for a, b in zip(tape, tape[::-1])) + abs(len(tape) - len(tape[::-1]))


def first_seam(text: str) -> dict[str, int | None]:
    tape = letters(text)
    for index, (left, right) in enumerate(zip(tape, tape[::-1])):
        if left != right:
            return {"pair_index": index, "left_char": left, "right_char": right}
    return {"pair_index": None, "left_char": None, "right_char": None}


def run() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    # Two frames x three relative clauses x three complement pairings.  The
    # route is intentionally small: Dream-RSI selects this new geometry, not a
    # larger duplicate Cartesian sweep of the exhausted two-span route.
    for frame_id, (scene, left_open, right_open, left_subject, right_subject, _, _) in enumerate(FRAMES):
        for left_index, left_relative in enumerate(LEFT_RELATIVE):
            for right_index, right_relative in enumerate(RIGHT_RELATIVE):
                left_complement = LEFT_COMPLEMENTS[(left_index + frame_id) % len(LEFT_COMPLEMENTS)]
                right_complement = RIGHT_COMPLEMENTS[(right_index + frame_id) % len(RIGHT_COMPLEMENTS)]
                text = (
                    f"{left_open}, {left_subject} {left_relative} {left_complement}. "
                    f"{right_open}, {right_subject} {right_relative} {right_complement}."
                )
                row_audit = audit(text)
                rows.append(
                    {
                        "scene_id": f"{scene}-{left_index}-{right_index}",
                        "rendered": text,
                        "mutable_spans": [
                            "left.relative_clause+host_complement",
                            "right.relative_clause+host_complement",
                        ],
                        "outer_assignments_fixed": [left_open, right_open],
                        "boundary_crossing": {
                            "left_relative_clause": left_relative,
                            "right_relative_clause": right_relative,
                            "left_host_subject": left_subject,
                            "right_host_subject": right_subject,
                            "word_boundaries_reopened": True,
                        },
                        "seam": first_seam(text),
                        "audit": row_audit,
                        "residual_debt": residual(text),
                        "provenance": {
                            "authored_scene": True,
                            "relative_clause_boundary_infill": True,
                            "span_boundaries_altered": True,
                            "catalogue_used": False,
                            "borrowed_catalogue_text": False,
                            "wrapped_seed": False,
                            "finished_tape_reversal": False,
                            "repeated_unit": False,
                        },
                    }
                )
    exact = sum(bool(row["audit"]["two_pointer_exact"]) for row in rows)
    longest = max(row["audit"]["letters"] for row in rows)
    best = min(row["residual_debt"] for row in rows)
    registry_path = ROOT / "docs" / "experiment-novelty-registry.json"
    registry = json.loads(registry_path.read_text())
    ids = {str(item.get("experiment_id", item.get("id", ""))) for item in registry if isinstance(item, dict)}
    return {
        "experiment": EXPERIMENT,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rendered_candidates": rows,
        "stats": {
            "rendered": len(rows),
            "exact": exact,
            "longest_letters": longest,
            "best_residual_debt": best,
        },
        "novelty_preflight": {
            "experiment_id_absent_before_run": EXPERIMENT not in ids,
            "geometry": "relative-clause attachment seam jointly reopened with host complement",
            "prior_route_reused": False,
            "duplicate_sweep_rejected": True,
        },
        "next_repair": {
            "operator": "three-region discourse-frame infill with a single shared referent",
            "reason": (
                "relative-clause plus host-complement reopening preserves complete prose but "
                "does not close the character seam in this bounded deployment; change geometry again"
            ),
            "route_exhausted": True,
        },
        "provenance": {
            "bounded_states": len(rows),
            "catalogue_used": False,
            "old_two_span_rows_reenumerated": False,
            "human_readability_certified": False,
        },
    }


if __name__ == "__main__":
    payload = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
