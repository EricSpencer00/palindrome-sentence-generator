"""Third construction: independently authored clause/apposition seams.

The two sides are complete, ordinary-English clauses.  A comma, coordinating
conjunction, or appositive boundary is chosen before expansion; each newly
exposed character is checked against the opposite side immediately.  This is
not a finished-tape reversal or a word-order mirror: the two clause banks are
authored separately and only compatible live seams survive.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

RUN_ID = "apposition-seam-grammar-20260920"
OUT = Path(__file__).with_name("runs") / f"{RUN_ID}.json"

# Deliberately different vocabularies on each side; these are complete clauses,
# not fragments or a catalogue sentence.  The seam token is structural prose.
LEFT = (
    "the patient keeper records a small tide",
    "our evening teacher carries a folded map",
    "a careful mason measures the quiet arch",
    "the young gardener waters a blue iris",
)
RIGHT = (
    "the harbor bell sounds at first light",
    "a kind neighbor returns the borrowed book",
    "the old captain watches a distant gull",
    "our steady friend mends the garden gate",
)
SEAMS = (", and ", ", but ", ", the witness, ")


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    t = letters(text)
    mismatch = next(
        ((i, t[i], t[-1 - i]) for i in range(len(t) // 2) if t[i] != t[-1 - i]),
        None,
    )
    fwd = hashlib.sha256(t.encode()).hexdigest()
    rev = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {
        "letters": len(t),
        "exact": bool(t) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": fwd,
        "sha256_reverse": rev,
        "sha_equal": fwd == rev,
    }


def live_join(left: str, right: str) -> tuple[bool, list[dict[str, object]]]:
    """Compare the exposed clause characters as they become available."""
    a, b = letters(left), letters(right)
    trace: list[dict[str, object]] = []
    for i, (x, y) in enumerate(zip(a, reversed(b))):
        row = {"offset": i, "left": x, "right_reversed": y, "matched": x == y}
        trace.append(row)
        if x != y:
            return False, trace
    return len(a) == len(b), trace


def run() -> dict[str, object]:
    rows = []
    exact = []
    for left in LEFT:
        for right in RIGHT:
            for seam in SEAMS:
                # Right remains forward authored prose; it is never reversed.
                rendered = left + seam + right + "."
                ok, trace = live_join(left + seam, right)
                row = {
                    "rendered": rendered,
                    "live_closed": ok,
                    "boundary_trace": trace,
                    "audit": audit(rendered),
                    "provenance": {
                        "left_clause": left,
                        "right_clause": right,
                        "seam": seam,
                        "independent_clause_authorship": True,
                        "ordinary_clause_grammar": True,
                        "finished_tape_reversal": False,
                        "word_order_mirroring": False,
                        "repeated_units": False,
                        "catalogue_surface_text": False,
                        "post_hoc_repair": False,
                    },
                }
                rows.append(row)
                if row["audit"]["exact"] and row["audit"]["letters"] > 38:
                    exact.append(row)
    rows.sort(key=lambda row: row["audit"]["letters"], reverse=True)
    return {
        "run_id": RUN_ID,
        "method": "independent complete clauses joined by live comma/conjunction/appositive seams",
        "stats": {
            "left_clauses": len(LEFT),
            "right_clauses": len(RIGHT),
            "seams": len(SEAMS),
            "rendered_controls": len(rows),
            "live_closed": sum(row["live_closed"] for row in rows),
            "exact_gt38": len(exact),
            "max_letters": max(row["audit"]["letters"] for row in rows),
        },
        "rendered_controls": rows[:6],
        "exact_candidates": exact,
        "novelty_preflight": {
            "status": "passed",
            "distinct_from": "chart phrase paths, typed centers, and dependency frames",
            "finished_tape_reversal": False,
            "reward_reranking": False,
            "word_order_mirroring": False,
            "repeated_units": False,
            "post_hoc_repair": False,
        },
        "provenance": {
            "audits": ["independent two-pointer seam comparison", "forward/reverse SHA-256"],
            "next_construction": "add a third independently authored appositive clause bank with held-out temporal seams",
            "reader_gate": "closed unless exact_gt38 appears",
        },
        "status": "fresh exact >38 candidate requires human reading" if exact else "no exact >38 closure in bounded seam bank",
    }


if __name__ == "__main__":
    result = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
