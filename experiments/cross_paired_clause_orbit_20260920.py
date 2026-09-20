"""Cross-paired two-clause construction with a live character seam.

Unlike a mirrored slot path, this lane pairs complete semantic clauses whose
roles are deliberately permuted across the centre.  Left and right phrases
are selected together and matched incrementally; no completed tape is
reversed, repaired, or replayed from a catalogue.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "cross-paired-clause-orbit-20260920.json"
EXPERIMENT_ID = "cross-paired-clause-orbit-20260920"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    mismatches = [(i, len(tape) - 1 - i) for i in range(len(tape) // 2)
                  if tape[i] != tape[-1 - i]]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": bool(tape) and not mismatches,
            "first_mismatch": mismatches[0] if mismatches else None,
            "sha256_forward": forward, "sha256_reverse": reverse,
            "sha_equal": forward == reverse}


def consume(left: str, right: str) -> tuple[str, str] | None:
    """Consume matched seam characters; retain only unmatched debt."""
    n = min(len(left), len(right))
    if n and left[:n] != right[-n:][::-1]:
        return None
    return left[n:], right[:-n] if n else right


@dataclass(frozen=True)
class Frame:
    role: str
    text: str
    valency: str
    split: str


def frame(role: str, valency: str, split: str, *texts: str) -> tuple[Frame, ...]:
    return tuple(Frame(role, text, valency, split) for text in texts)


def build_lattice() -> tuple[tuple[Frame, ...], ...]:
    # Clause A: subject -> event -> object -> setting.
    # Clause B is deliberately role-permuted: setting -> object -> event ->
    # subject.  The alternatives are authored ordinary prose, not corpus
    # sentences.  The run uses the held-out B valency bank at the right edge.
    subject = frame("subject", "agent", "A", "the bard", "the keeper", "a singer", "my lord")
    event = frame("event", "transitive", "A", "reads the letter", "keeps the oath", "sees the moon", "speaks the truth")
    obj = frame("object", "patient", "A", "the old book", "a bright sign", "the last note", "the red rose")
    setting = frame("setting", "locative", "A", "in the hall", "at dawn", "by the fire", "under the moon")
    # Held-out lexicalizations have different valency heads and are not
    # aliases of the A-side phrase bank.
    held_setting = frame("setting", "locative-heldout", "B", "beneath the stars", "beside the gate", "within the court")
    held_obj = frame("object", "patient-heldout", "B", "a sealed scroll", "the black seal", "one lost vow")
    held_event = frame("event", "ditransitive-heldout", "B", "carries the word", "guards the truth", "names the heir")
    held_subject = frame("subject", "agent-heldout", "B", "the queen", "an old friend", "our singer")
    return (subject, event, obj, setting, held_setting, held_obj, held_event, held_subject)


def run(*, state_limit: int = 300_000, witness_limit: int = 24) -> dict[str, object]:
    lattice = build_lattice()
    states = pruned = orbit_steps = 0
    witnesses: list[dict[str, object]] = []
    exact_rows: list[dict[str, object]] = []

    def record_witness(words: tuple[Frame, ...], left: str, right: str, depth: int) -> None:
        if len(witnesses) >= witness_limit:
            return
        rendered = " ".join(x.text for x in words)
        witnesses.append({"rendered": rendered, "depth": depth,
                          "left_residual": len(left), "right_residual": len(right),
                          "audit": audit(rendered),
                          "reader_status": "diagnostic witness; not a complete candidate"})

    def walk(lo: int, hi: int, left: str, right: str,
             lf: tuple[Frame, ...], rf: tuple[Frame, ...]) -> None:
        nonlocal states, pruned, orbit_steps
        if states >= state_limit:
            return
        if lo > hi:
            if left or right:
                return
            ordered = lf + tuple(reversed(rf))
            rendered = " ".join(x.text for x in ordered)
            checked = audit(rendered)
            if checked["exact"]:
                exact_rows.append({"rendered": rendered, "audit": checked,
                    "provenance": {"construction": "cross-paired complete clause orbit",
                        "roles": [x.role for x in ordered],
                        "valencies": [x.valency for x in ordered],
                        "held_out_right_clause": True,
                        "finished_tape_reversal": False, "post_hoc_repair": False,
                        "catalogue_text": False, "aligned_token_mirror": False},
                    "reader_status": "unreviewed; exactness does not certify readability"})
            return
        if lo == hi:
            for f in lattice[lo]:
                states += 1
                nl = left + letters(f.text)
                residual = consume(nl, right)
                if residual is None:
                    pruned += 1
                    record_witness(lf + (f,) + tuple(reversed(rf)), nl, right, len(lf) + 1)
                else:
                    orbit_steps += 1
                    walk(lo + 1, hi - 1, residual[0], residual[1], lf + (f,), rf)
            return
        for lf_frame in lattice[lo]:
            for rf_frame in lattice[hi]:
                states += 1
                nl = left + letters(lf_frame.text)
                nr = letters(rf_frame.text) + right
                residual = consume(nl, nr)
                if residual is None:
                    pruned += 1
                    record_witness(lf + (lf_frame,) + (rf_frame,) + tuple(reversed(rf)), nl, nr, len(lf) + 1)
                    continue
                orbit_steps += 1
                walk(lo + 1, hi - 1, residual[0], residual[1],
                     lf + (lf_frame,), (rf_frame,) + rf)

    walk(0, len(lattice) - 1, "", "", (), ())
    exact_rows.sort(key=lambda row: row["audit"]["letters"], reverse=True)
    result = {"experiment": EXPERIMENT_ID,
              "method": "cross-paired complete semantic clauses with live seam orbit",
              "candidates": exact_rows, "witnesses": witnesses,
              "stats": {"states": states, "pruned": pruned,
                        "orbit_steps": orbit_steps, "exact": len(exact_rows)},
              "provenance": {"complete_left_clause": True, "complete_right_clause": True,
                  "role_permutation": "A subject-event-object-setting / B setting-object-event-subject",
                  "held_out_valency_frames": True, "finished_tape_reversal": False,
                  "post_hoc_repair": False, "catalogue_text": False,
                  "aligned_token_mirror": False,
                  "next_construction": "add grammatical tense/agreement variants to the held-out clause while preserving cross-role order"}}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
