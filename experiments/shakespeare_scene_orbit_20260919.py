"""Joint scene-lattice construction with live character-orbit equations.

This lane treats a short Shakespearean scene as a typed sequence of semantic
frames.  A frame supplies a complete phrase (not a catalogue token), and the
left and right scene positions are selected together.  The only pruning rule
is the character equation induced by the already-built outer orbit; no
finished string is reversed and no repair is applied afterwards.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "shakespeare-scene-orbit-20260919.json"
EXPERIMENT_ID = "shakespeare-scene-orbit-20260919"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    fwd = hashlib.sha256(tape.encode()).hexdigest()
    rev = hashlib.sha256(tape[::-1].encode()).hexdigest()
    mismatches = [(i, len(tape) - 1 - i) for i in range(len(tape) // 2)
                  if tape[i] != tape[-1 - i]]
    return {"letters": len(tape), "exact": bool(tape) and not mismatches,
            "first_mismatch": mismatches[0] if mismatches else None,
            "sha256_forward": fwd, "sha256_reverse": rev,
            "sha_equal": fwd == rev}


@dataclass(frozen=True)
class Frame:
    role: str
    text: str
    scene: str
    valency: str


def consume(left: str, right: str) -> tuple[str, str] | None:
    """Consume the currently paired outer characters.

    ``left`` is ordered from the left seam inward.  ``right`` is ordered
    from the right seam inward, so its *end* is the next character paired
    with ``left[0]``.  Matched characters are removed immediately; retaining
    them was the bug in the first implementation and made the search reject
    valid paths after the first unequal word boundary.
    """
    n = min(len(left), len(right))
    if left[:n] != right[-n:][::-1] if n else False:
        return None
    return left[n:], right[:-n] if n else right


def _frame(role: str, scene: str, valency: str, *texts: str) -> tuple[Frame, ...]:
    return tuple(Frame(role, text, scene, valency) for text in texts)


def build_scene_lattice() -> tuple[tuple[Frame, ...], ...]:
    # These are authored, ordinary scene phrases with explicit argument
    # structure.  They are deliberately small: this is a construction audit,
    # not a corpus replay or a word list sweep.
    return (
        _frame("opening_np", "court", "subject", "the king", "the bard", "a prince", "my lord"),
        _frame("finite_event", "court", "transitive", "keeps the oath", "reads the letter", "sees the moon", "speaks the truth"),
        _frame("manner_pp", "court", "adjunct", "in the hall", "at dawn", "by the fire", "under the moon"),
        _frame("relative", "court", "modifier", "who hears", "that remembers", "who waits", "that sings"),
        _frame("finite_event", "court", "transitive", "keeps the oath", "reads the letter", "sees the moon", "speaks the truth"),
        _frame("closing_np", "court", "object", "the king", "the bard", "a prince", "my lord"),
    )


def run(*, state_limit: int = 250_000, candidate_limit: int = 32) -> dict[str, object]:
    lattice = build_scene_lattice()
    states = pruned = orbit_steps = 0
    candidates: list[dict[str, object]] = []
    witnesses: list[dict[str, object]] = []

    def add_witness(rendered: str, depth: int, left: str, right: str) -> None:
        if len(witnesses) < 20:
            witnesses.append({"rendered": rendered, "depth": depth,
                              "left_buffer": len(left), "right_buffer": len(right),
                              "audit": audit(rendered),
                              "reader_status": "diagnostic witness; not a complete candidate"})

    def walk(lo: int, hi: int, left: str, right: str,
             lf: tuple[Frame, ...], rf: tuple[Frame, ...]) -> None:
        nonlocal states, pruned, orbit_steps
        if states >= state_limit or len(candidates) >= candidate_limit:
            return
        if lo > hi:
            rendered = " ".join(f.text for f in lf + tuple(reversed(rf)))
            checked = audit(rendered)
            if checked["exact"] and len(set(f.text for f in lf + rf)) == len(lf) + len(rf):
                candidates.append({"rendered": rendered, "audit": checked,
                    "provenance": {"construction": "authored semantic scene orbit",
                        "roles": [f.role for f in lf + tuple(reversed(rf))],
                        "valencies": [f.valency for f in lf + tuple(reversed(rf))],
                        "scenes": [f.scene for f in lf + tuple(reversed(rf))],
                        "finished_tape_reversal": False, "post_hoc_repair": False,
                        "catalogue_text": False, "aligned_token_mirror": False},
                    "reader_status": "unreviewed; exactness does not certify readability"})
            return
        if lo == hi:
            for f in lattice[lo]:
                nl = left + letters(f.text)
                states += 1
                residual = consume(nl, right)
                if residual is not None:
                    orbit_steps += 1
                    walk(lo + 1, hi - 1, residual[0], residual[1], lf + (f,), rf)
                else:
                    pruned += 1
                    add_witness(" ".join(x.text for x in lf + (f,) + tuple(reversed(rf))), len(lf) + 1, nl, right)
            return
        for lframe in lattice[lo]:
            for rframe in lattice[hi]:
                states += 1
                nl = left + letters(lframe.text)
                nr = letters(rframe.text) + right
                # The orbit equation checks every currently paired endpoint;
                # unequal buffers are intentionally retained, not indexed by
                # the newest word or repaired later.
                residual = consume(nl, nr)
                if residual is None:
                    pruned += 1
                    add_witness(" ".join(x.text for x in lf + (lframe,) + (rframe,) + tuple(reversed(rf))), len(lf) + 1, nl, nr)
                    continue
                orbit_steps += 1
                walk(lo + 1, hi - 1, residual[0], residual[1],
                     lf + (lframe,), (rframe,) + rf)

    walk(0, len(lattice) - 1, "", "", (), ())
    candidates.sort(key=lambda x: x["audit"]["letters"], reverse=True)
    result = {"experiment": EXPERIMENT_ID,
              "method": "authored semantic scene lattice with live character orbit",
              "candidates": candidates, "witnesses": witnesses,
              "stats": {"states": states, "pruned": pruned,
                        "orbit_steps": orbit_steps, "exact": len(candidates)},
              "provenance": {"human_authored_frames": True,
                  "corpus_replay": False, "finished_tape_reversal": False,
                  "post_hoc_repair": False, "catalogue_text": False,
                  "aligned_token_mirror": False,
        "next_construction": "add an authored two-clause scene with explicit subject/object valency and retain consumed residual buffers",
        "implementation_audit": "corrected after initial run: every orbit step now consumes the matched overlap; the pre-correction 16-state result is invalid"}}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
