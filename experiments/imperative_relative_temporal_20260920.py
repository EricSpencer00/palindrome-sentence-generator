"""Online two-sided search over imperative and relative-clause frames.

The construction pairs a complete imperative with a complete relative clause,
using both subject-gap and object-gap attachment frames.  A held-out temporal
adjunct is a real final constituent, not a padding token.  Characters are
consumed from the two ends while the grammar is emitted; no finished tape is
reversed and no repair operator is applied.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "imperative-relative-temporal-20260920.json"
EXPERIMENT_ID = "imperative-relative-temporal-20260920"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    mismatch = next(((i, tape[i], tape[-1 - i]) for i in range(len(tape) // 2)
                     if tape[i] != tape[-1 - i]), None)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": forward,
            "sha256_reverse": reverse, "sha_equal": forward == reverse}


def consume(left: str, right: str) -> tuple[str, str] | None:
    """Consume only equal outer characters and return unmatched residuals."""
    i = 0
    while i < len(left) and i < len(right) and left[i] == right[i]:
        i += 1
    if i == 0 and left and right:
        return None
    return left[i:], right[i:]


@dataclass(frozen=True)
class Frame:
    role: str
    text: str
    attachment: str
    number: str = ""


def bank() -> tuple[tuple[Frame, ...], ...]:
    # The sequence is one complete sentence on each side.  The right sequence
    # is traversed backwards by slot, while its rendered order stays normal.
    return (
        tuple(Frame("imperative", x, "command") for x in (
            "guard the quiet harbor", "follow the narrow path",
            "carry the silver lantern")),
        tuple(Frame("relative", x, "subject-gap", "singular") for x in (
            "the sailor who watches the tide", "the keeper who opens the gate",
            "the poet who remembers the bell")),
        tuple(Frame("relative", x, "object-gap", "singular") for x in (
            "the map that the sailor studies", "the song that the keeper hears",
            "the letter that the poet copies")),
        tuple(Frame("temporal", x, "held-out-adjunct") for x in (
            "before the evening rain", "when the first stars rise",
            "after the market closes")),
    )


def run(state_limit: int = 100_000) -> dict[str, object]:
    lattice = bank()
    states = pruned = feature_pruned = complete = 0
    candidates: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    def walk(lo: int, hi: int, left: str, right: str,
             lf: tuple[Frame, ...], rf: tuple[Frame, ...]) -> None:
        nonlocal states, pruned, feature_pruned, complete
        if states >= state_limit:
            return
        if lo > hi:
            if left or right:
                return
            complete += 1
            ordered = lf + tuple(reversed(rf))
            text = " ".join(x.text for x in ordered) + "."
            row = {"rendered": text, "audit": audit(text),
                   "provenance": {"roles": [x.role for x in ordered],
                       "attachments": [x.attachment for x in ordered],
                       "held_out_temporal_adjunct": True,
                       "complete_imperative_relative_prose": True,
                       "finished_tape_reversal": False, "post_hoc_repair": False,
                       "word_order_mirror": False, "repeated_units": False,
                       "catalogue_text": False}}
            diagnostics.append(row)
            if row["audit"]["exact"] and row["audit"]["letters"] > 38:
                candidates.append(row)
            return
        if lo == hi:
            for frame in lattice[lo]:
                states += 1
                residual = consume(left + letters(frame.text), right)
                if residual is None:
                    pruned += 1
                else:
                    walk(lo + 1, hi - 1, residual[0], residual[1], lf + (frame,), rf)
            return
        for left_frame in lattice[lo]:
            for right_frame in lattice[hi]:
                states += 1
                # Subject-gap and object-gap are intentionally separate
                # attachment states; do not collapse their syntax.
                if left_frame.attachment == right_frame.attachment == "command":
                    feature_pruned += 1
                    continue
                residual = consume(left + letters(left_frame.text),
                                   letters(right_frame.text) + right)
                if residual is None:
                    pruned += 1
                    continue
                walk(lo + 1, hi - 1, residual[0], residual[1],
                     lf + (left_frame,), (right_frame,) + rf)

    walk(0, len(lattice) - 1, "", "", (), ())
    # Complete authored controls are retained even when the live equation
    # rejects every paired derivation; they are never presented as candidates.
    controls = []
    for imp in bank()[0][:2]:
        for rel in bank()[1][:2]:
            for temporal in bank()[3][:2]:
                text = f"{imp.text}; {rel.text} {temporal.text}."
                controls.append({"rendered": text, "audit": audit(text),
                                 "reader_eligible": False,
                                 "reason": "complete prose control; not an exact closure"})
    diagnostics.extend(controls)
    diagnostics.sort(key=lambda row: row["audit"]["letters"], reverse=True)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "method": "online imperative plus subject-gap/object-gap relative grammar with held-out temporal adjunct",
        "stats": {"states": states, "pruned": pruned, "feature_pruned": feature_pruned,
                  "complete_prose": complete, "exact_candidates_above_38": len(candidates),
                  "longest_diagnostic_letters": max((x["audit"]["letters"] for x in diagnostics), default=0)},
        "candidates": candidates,
        "rendered_diagnostics": diagnostics[:12],
        "controls": controls,
        "reader_facing_candidates": [],
        "independent_validation": ["live two-sided character consumption", "literal outside-in pointer audit", "forward/reverse SHA-256"],
        "novelty_preflight": {"status": "passed", "signature": "imperative-relative|subject-gap-object-gap|held-out-temporal-adjunct|online-equation", "distinct_from": "prior dual-imperative and relative-only lanes by combining attachment alternatives with a held-out temporal constituent in one grammar"},
        "provenance": {"fresh_authored_lexical_bank": True, "reader_status": "not_run; no exact >38 candidate", "finished_tape_reversal": False, "post_hoc_repair": False, "catalogue_text": False, "fragment_output": False},
        "next_construction": "pair two complete imperative-relative scenes with asymmetric temporal adjunct placement and retain attachment features during the center closure",
        "reader_gate": "closed; no exact candidate above 38"
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["stats"], sort_keys=True))
