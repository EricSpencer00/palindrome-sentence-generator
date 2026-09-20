"""Dual relative-attachment construction with valency-conditioned seams."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "dual-relative-valency-orbit-20260920.json"
EXPERIMENT_ID = "dual-relative-valency-orbit-20260920"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    mismatches = [(i, len(tape) - i - 1) for i in range(len(tape) // 2)
                  if tape[i] != tape[-i - 1]]
    fwd = hashlib.sha256(tape.encode()).hexdigest()
    rev = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": bool(tape) and not mismatches,
            "first_mismatch": mismatches[0] if mismatches else None,
            "sha256_forward": fwd, "sha256_reverse": rev,
            "sha_equal": fwd == rev}


def consume(left: str, right: str) -> tuple[str, str] | None:
    n = min(len(left), len(right))
    if n and left[:n] != right[-n:][::-1]:
        return None
    return left[n:], right[:-n] if n else right


@dataclass(frozen=True)
class Frame:
    role: str
    text: str
    clause: str
    valency: str
    number: str | None = None
    agreement: str | None = None
    tense: str | None = None


def fs(role: str, clause: str, valency: str, *texts: str,
       number: str | None = None, agreement: str | None = None,
       tense: str | None = None) -> tuple[Frame, ...]:
    return tuple(Frame(role, text, clause, valency, number, agreement, tense) for text in texts)


def build_lattice() -> tuple[tuple[Frame, ...], ...]:
    # Rendered grammar:
    #   main subject [subject-relative] event object while
    #   subordinate subject event object [object-relative]
    # Pair order lists the right side from the outside inward.
    return (
        fs("subject", "main", "agent", "the steward", "a raven", "the poet", number="singular"),
        fs("relative_subject", "main-relative", "subject-modifier", "who waits", "that sings", "who remembers", agreement="singular", tense="present"),
        fs("event", "main", "transitive", "keeps the vow", "marks the page", "kept the vow", agreement="singular", tense="present"),
        fs("object", "main", "patient", "a silver cup", "the quiet bell", "one true word"),
        fs("subordinator", "bridge", "conjunction", "while", "though", "when"),
        fs("relative_object", "subordinate-relative", "object-modifier", "that they found", "which they kept", "that we carried", agreement="plural", tense="past"),
        fs("object", "subordinate", "patient", "the winter seal", "a hidden key", "one pale crown"),
        fs("event", "subordinate", "transitive", "guard the gate", "name the heir", "guarded the gate", agreement="plural", tense="present"),
        fs("subject", "subordinate", "agent", "the guards", "some friends", "the queens", number="plural"),
    )


def run(*, state_limit: int = 250_000) -> dict[str, object]:
    lattice = build_lattice()
    states = pruned = feature_pruned = orbit_steps = 0
    candidates: list[dict[str, object]] = []
    witnesses: list[dict[str, object]] = []

    def witness(words: tuple[Frame, ...], left: str, right: str, depth: int, reason: str) -> None:
        if len(witnesses) < 24:
            rendered = " ".join(x.text for x in words)
            witnesses.append({"rendered": rendered, "depth": depth, "reason": reason,
                              "left_residual": len(left), "right_residual": len(right),
                              "audit": audit(rendered),
                              "reader_status": "diagnostic witness; not a complete candidate"})

    def walk(lo: int, hi: int, left: str, right: str,
             lf: tuple[Frame, ...], rf: tuple[Frame, ...], state: dict[str, str]) -> None:
        nonlocal states, pruned, feature_pruned, orbit_steps
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
                    "provenance": {"construction": "dual relative attachments with valency-conditioned seam",
                        "roles": [x.role for x in ordered],
                        "clauses": [x.clause for x in ordered],
                        "valencies": [x.valency for x in ordered],
                        "features": [{"number": x.number, "agreement": x.agreement,
                                      "tense": x.tense} for x in ordered],
                        "held_out_subordinate_lexicalizations": True,
                        "finished_tape_reversal": False, "post_hoc_repair": False,
                        "catalogue_text": False, "aligned_token_mirror": False},
                    "reader_status": "unreviewed; exactness does not certify readability"})
            return
        if lo == hi:
            for x in lattice[lo]:
                states += 1
                residual = consume(left + letters(x.text), right)
                if residual is None:
                    pruned += 1
                    witness(lf + (x,) + tuple(reversed(rf)), left + letters(x.text), right, len(lf) + 1, "character seam")
                else:
                    orbit_steps += 1
                    walk(lo + 1, hi - 1, residual[0], residual[1], lf + (x,), rf, dict(state))
            return
        for lframe in lattice[lo]:
            for rframe in lattice[hi]:
                states += 1
                next_state = dict(state)
                if lframe.role == "subject":
                    next_state["main_number"] = lframe.number or ""
                if rframe.role == "subject":
                    next_state["sub_number"] = rframe.number or ""
                if lframe.role in {"event", "relative_subject"} and lframe.agreement != next_state.get("main_number"):
                    feature_pruned += 1
                    continue
                if rframe.role in {"event", "relative_object"} and rframe.agreement not in {None, "plural"}:
                    feature_pruned += 1
                    continue
                residual = consume(left + letters(lframe.text), letters(rframe.text) + right)
                if residual is None:
                    pruned += 1
                    witness(lf + (lframe,) + (rframe,) + tuple(reversed(rf)), left + letters(lframe.text), letters(rframe.text) + right, len(lf) + 1, "character seam")
                    continue
                orbit_steps += 1
                walk(lo + 1, hi - 1, residual[0], residual[1], lf + (lframe,), (rframe,) + rf, next_state)

    walk(0, len(lattice) - 1, "", "", (), (), {})
    candidates.sort(key=lambda row: row["audit"]["letters"], reverse=True)
    result = {"experiment": EXPERIMENT_ID,
              "method": "dual relative attachments with valency-conditioned live seam",
              "candidates": candidates, "witnesses": witnesses,
              "stats": {"states": states, "pruned": pruned,
                        "feature_pruned": feature_pruned, "orbit_steps": orbit_steps,
                        "exact": len(candidates)},
              "provenance": {"complete_main_clause": True, "complete_subordinate_clause": True,
                  "two_relative_attachment_sites": True, "valency_conditioned": True,
                  "agreement_state_before_emission": True,
                  "held_out_subordinate_lexicalizations": True,
                  "finished_tape_reversal": False, "post_hoc_repair": False,
                  "catalogue_text": False, "aligned_token_mirror": False,
                  "next_construction": "test a relative clause attached to the bridge conjunction with explicit complement selection"}}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
