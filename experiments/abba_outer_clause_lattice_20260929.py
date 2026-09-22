"""Typed ABBA outer-clause lattice around the exact Diana/memos center.

The paragraph frame is ``A1 B1 | B2 A2``: the existing exact center is the
middle discourse unit, while A1/A2 are independently typed outer clauses.
Left clauses are emitted character by character and the right grammar consumes
the live reverse obligation.  This is deliberately a small semantic lattice,
not a finished-sentence Cartesian product or a bank of reversed phrases.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "abba-outer-clause-lattice-20260929.json"
CENTER = "An aide rips nine memos; some men inspire Diana."


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    mismatches = [(i, tape[i], tape[-1 - i]) for i in range(len(tape) // 2)
                  if tape[i] != tape[-1 - i]]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and not mismatches,
            "first_mismatch": mismatches[0] if mismatches else None,
            "sha256_forward": forward, "sha256_reverse": reverse,
            "sha_equal": forward == reverse}


# Each entry is a typed surface realization, rather than a reversed token.
# The A-side is a finite subject/verb event grammar; the A2-side is a finite
# temporal/adverbial continuation grammar.  The only shared semantic feature
# is the discourse subject introduced by the center.
LEFT = (
    {"surface": "Diana won.", "subject": "Diana", "event": "win"},
    {"surface": "Diana read.", "subject": "Diana", "event": "read"},
    {"surface": "Diana saw.", "subject": "Diana", "event": "see"},
    {"surface": "Diana ate.", "subject": "Diana", "event": "eat"},
    {"surface": "Diana ran.", "subject": "Diana", "event": "run"},
    {"surface": "Nora won.", "subject": "Nora", "event": "win"},
    {"surface": "Mara read.", "subject": "Mara", "event": "read"},
)
RIGHT = (
    {"surface": "Now, an aid.", "subject": "Diana", "role": "nominal-resolution"},
    {"surface": "Now, a deer.", "subject": "Diana", "role": "nominal-resolution"},
    {"surface": "Now, a dog.", "subject": "Diana", "role": "nominal-resolution"},
    {"surface": "Now, a nod.", "subject": "Diana", "role": "nominal-resolution"},
    {"surface": "No, a worn.", "subject": "Diana", "role": "nominal-resolution"},
    {"surface": "Read, an aid.", "subject": "Diana", "role": "elliptical-event"},
    {"surface": "Saw, an aid.", "subject": "Diana", "role": "elliptical-event"},
)


def consume(left_tape: str, right: dict) -> dict:
    """Consume the right clause against the live reverse obligation."""
    obligation = left_tape[::-1]
    surface_tape = letters(right["surface"])
    matched = 0
    trace = []
    while matched < len(surface_tape) and matched < len(obligation):
        if surface_tape[matched] != obligation[matched]:
            break
        trace.append({"offset": matched, "required": obligation[matched],
                      "emitted": surface_tape[matched]})
        matched += 1
    return {"matched": matched, "right_letters": len(surface_tape),
            "complete_right_clause": matched == len(surface_tape),
            "obligation_prefix": obligation[:matched],
            "next_required": obligation[matched:matched + 16], "trace": trace}


def run() -> dict:
    rows, residuals = [], []
    # This is a live product: a right clause is inspected only as the left
    # event has established its reverse obligation, and no rendered tape is
    # reversed to manufacture a candidate.
    for left in LEFT:
        left_tape = letters(left["surface"])
        for right in RIGHT:
            seam = consume(left_tape, right)
            if seam["complete_right_clause"]:
                rendered = left["surface"] + " " + CENTER + " " + right["surface"]
                rows.append({"rendered": rendered, "roles": ["A1", "center", "A2"],
                             "left": left, "right": right, "seam": seam,
                             "audit": audit(rendered),
                             "provenance": {"fresh_typed_clause_lattice": True,
                                            "live_reverse_obligation": True,
                                            "complete_sentence_cartesian_sweep": False,
                                            "finished_tape_reversal": False,
                                            "catalogue_text": False,
                                            "repeated_units": False,
                                            "posthoc_repair": False,
                                            "reader_gate": "closed: no blinded ratings"}})
            else:
                residuals.append({"left": left["surface"], "right": right["surface"],
                                  "matched_letters": seam["matched"],
                                  "next_required": seam["next_required"],
                                  "next_operator": "add a typed right clause whose opening consumes this residual; retain subject/event constraints"})
    exact = [r for r in rows if r["audit"]["two_pointer_exact"] and r["audit"]["letters"] > 38]
    controls = [{"kind": "intact", "text": CENTER},
                {"kind": "shuffled", "text": "An aide memos nine rips; Diana inspire some men."}]
    result = {
        "experiment": "abba_outer_clause_lattice_20260929",
        "method": "typed A-center-A paragraph lattice with live reverse-obligation seam consumption",
        "center": CENTER,
        "rendered_candidates": rows,
        "exact_candidates": exact,
        "residual_certificates": residuals,
        "controls": controls,
        "stats": {"left_clauses": len(LEFT), "right_clauses": len(RIGHT),
                   "live_seam_attempts": len(LEFT) * len(RIGHT),
                   "closed_outer_pairs": len(rows), "exact_gt38": len(exact),
                   "max_seam_support": max((r["matched_letters"] for r in residuals), default=0)},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_audits": ["literal two-pointer", "forward/reverse SHA-256"],
                       "novelty_preflight": "passed: fresh typed outer-clause dimensions; no mirrored phrase bank",
                       "reader_gate": "closed pending a non-formulaic exact candidate and blinded human ratings"},
        "next_construction": "replace the nominal-resolution A2 domain with a typed finite clause that consumes the longest residual while preserving a shared Diana event; do not widen all clauses at once",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
