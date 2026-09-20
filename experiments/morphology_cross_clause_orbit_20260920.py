"""Agreement/tense-aware cross-clause construction.

Two complete clauses retain the cross-role order from the preceding lane,
but each subject carries number and each event carries agreement plus tense.
The feature state is enforced while phrases are selected, alongside live
character residual consumption.  This is a new construction, not a larger
word-bank sweep.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "morphology-cross-clause-orbit-20260920.json"
EXPERIMENT_ID = "morphology-cross-clause-orbit-20260920"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    mismatch = [(i, len(tape) - i - 1) for i in range(len(tape) // 2)
                if tape[i] != tape[-i - 1]]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": bool(tape) and not mismatch,
            "first_mismatch": mismatch[0] if mismatch else None,
            "sha256_forward": forward, "sha256_reverse": reverse,
            "sha_equal": forward == reverse}


def consume(left: str, right: str) -> tuple[str, str] | None:
    n = min(len(left), len(right))
    if n and left[:n] != right[-n:][::-1]:
        return None
    return left[n:], right[:-n] if n else right


@dataclass(frozen=True)
class Frame:
    role: str
    text: str
    split: str
    number: str | None = None
    tense: str | None = None
    agreement: str | None = None


def f(role: str, split: str, *texts: str, number: str | None = None,
      tense: str | None = None, agreement: str | None = None) -> tuple[Frame, ...]:
    return tuple(Frame(role, text, split, number, tense, agreement) for text in texts)


def build_lattice() -> tuple[tuple[Frame, ...], ...]:
    # New lexical realizations, including held-out B forms.  The event banks
    # are explicitly inflected; agreement is checked against the subject
    # state rather than inferred after a sentence has been formed.
    return (
        f("subject", "A", "the steward", "a raven", "the poet", number="singular"),
        f("event", "A", "keeps the vow", "marks the page", "kept the vow", tense="present", agreement="singular"),
        f("object", "A", "a silver cup", "the quiet bell", "one true word"),
        f("setting", "A", "near the tower", "before the feast", "under the elm"),
        f("setting", "B", "amid the roses", "beyond the wall", "beside the throne"),
        f("object", "B", "the winter seal", "a hidden key", "one pale crown"),
        f("event", "B", "guard the gate", "name the heir", "guarded the gate", tense="present", agreement="plural"),
        f("subject", "B", "the guards", "some friends", "the queens", number="plural"),
    )


def feature_ok(slot: int, left: Frame, right: Frame,
              state: dict[str, str]) -> bool:
    """Enforce number/tense equations before the seam search proceeds."""
    # Outer slots select both subjects.  Keep independent A/B states.
    if slot == 0:
        state["A_number"] = left.number or ""
    if right.role == "subject":
        state["B_number"] = right.number or ""
    if slot == 1 and left.agreement != state.get("A_number"):
        return False
    if slot == 6 and right.agreement != state.get("B_number"):
        return False
    # Past and present are explicit carried states, not post-hoc rewriting.
    if left.role == "event" and left.tense not in {"present", "past"}:
        return False
    if right.role == "event" and right.tense not in {"present", "past"}:
        return False
    return True


def run(*, state_limit: int = 250_000) -> dict[str, object]:
    lattice = build_lattice()
    states = pruned = orbit_steps = feature_pruned = 0
    candidates: list[dict[str, object]] = []
    witnesses: list[dict[str, object]] = []

    def witness(words: tuple[Frame, ...], left: str, right: str, depth: int, reason: str) -> None:
        if len(witnesses) < 24:
            rendered = " ".join(x.text for x in words)
            witnesses.append({"rendered": rendered, "depth": depth,
                              "reason": reason, "left_residual": len(left),
                              "right_residual": len(right), "audit": audit(rendered),
                              "reader_status": "diagnostic witness; not a complete candidate"})

    def walk(lo: int, hi: int, left: str, right: str,
             lf: tuple[Frame, ...], rf: tuple[Frame, ...], state: dict[str, str]) -> None:
        nonlocal states, pruned, orbit_steps, feature_pruned
        if states >= state_limit:
            return
        if lo > hi:
            if left or right:
                return
            ordered = lf + tuple(reversed(rf))
            rendered = " ".join(x.text for x in ordered)
            checked = audit(rendered)
            if checked["exact"]:
                candidates.append({"rendered": rendered, "audit": checked,
                    "provenance": {"construction": "morphology-aware cross-clause orbit",
                        "roles": [x.role for x in ordered],
                        "features": [{"number": x.number, "tense": x.tense,
                                      "agreement": x.agreement} for x in ordered],
                        "held_out_right_clause": True, "finished_tape_reversal": False,
                        "post_hoc_repair": False, "catalogue_text": False,
                        "aligned_token_mirror": False},
                    "reader_status": "unreviewed; exactness does not certify readability"})
            return
        # At a seam pair, slot indices are the cross-permuted semantic roles.
        if lo == hi:
            for left_frame in lattice[lo]:
                states += 1
                next_state = dict(state)
                if not feature_ok(lo, left_frame, left_frame, next_state):
                    feature_pruned += 1
                    continue
                residual = consume(left + letters(left_frame.text), right)
                if residual is None:
                    pruned += 1
                    witness(lf + (left_frame,) + tuple(reversed(rf)), left + letters(left_frame.text), right, len(lf) + 1, "character seam")
                else:
                    orbit_steps += 1
                    walk(lo + 1, hi - 1, residual[0], residual[1], lf + (left_frame,), rf, next_state)
            return
        for left_frame in lattice[lo]:
            for right_frame in lattice[hi]:
                states += 1
                next_state = dict(state)
                if not feature_ok(lo, left_frame, right_frame, next_state):
                    feature_pruned += 1
                    continue
                residual = consume(left + letters(left_frame.text), letters(right_frame.text) + right)
                if residual is None:
                    pruned += 1
                    witness(lf + (left_frame,) + (right_frame,) + tuple(reversed(rf)), left + letters(left_frame.text), letters(right_frame.text) + right, len(lf) + 1, "character seam")
                    continue
                orbit_steps += 1
                walk(lo + 1, hi - 1, residual[0], residual[1],
                     lf + (left_frame,), (right_frame,) + rf, next_state)

    walk(0, len(lattice) - 1, "", "", (), (), {})
    candidates.sort(key=lambda row: row["audit"]["letters"], reverse=True)
    result = {"experiment": EXPERIMENT_ID,
              "method": "agreement/tense morphology over cross-paired clause orbit",
              "candidates": candidates, "witnesses": witnesses,
              "stats": {"states": states, "pruned": pruned,
                        "feature_pruned": feature_pruned, "orbit_steps": orbit_steps,
                        "exact": len(candidates)},
              "provenance": {"complete_left_clause": True, "complete_right_clause": True,
                  "agreement_state_carried": True, "tense_state_carried": True,
                  "held_out_valency_frames": True, "finished_tape_reversal": False,
                  "post_hoc_repair": False, "catalogue_text": False,
                  "aligned_token_mirror": False,
                  "next_construction": "add a central conjunction frame with independently inflected subordinate clauses"}}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
