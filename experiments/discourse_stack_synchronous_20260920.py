"""Discourse-conditioned synchronous grammar with open constituent stacks.

One semantic plan supplies shared referents and event features.  Two sides
realize that plan independently through typed stack transitions; only then do
their newly emitted character buffers enter the live palindrome equation.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "discourse-stack-synchronous-20260920.json"
EXPERIMENT_ID = "discourse-stack-synchronous-20260920"
SIGNATURE = "discourse-plan-conditioned|delayed-surface-realization|open-constituent-stack|shared-referents|attachment-before-output|joint-character-output"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    bad = [(i, len(tape) - i - 1) for i in range(len(tape) // 2)
           if tape[i] != tape[-i - 1]]
    fwd = hashlib.sha256(tape.encode()).hexdigest()
    rev = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": bool(tape) and not bad,
            "first_mismatch": bad[0] if bad else None,
            "sha256_forward": fwd, "sha256_reverse": rev,
            "sha_equal": fwd == rev}


def consume(left: str, right: str) -> tuple[str, str] | None:
    n = min(len(left), len(right))
    if n and left[:n] != right[-n:][::-1]:
        return None
    return left[n:], right[:-n] if n else right


@dataclass(frozen=True)
class Plan:
    plan_id: str
    subject_ref: str
    event_ref: str
    object_ref: str
    tense: str
    attachment: str


@dataclass(frozen=True)
class Realization:
    role: str
    text: str
    referent: str
    feature: str


def plans() -> tuple[Plan, ...]:
    return (
        Plan("dawn-letter", "mara", "read", "letter", "present", "pp"),
        Plan("gate-song", "noah", "open", "gate", "present", "adverb"),
        Plan("oath-book", "the-keeper", "guard", "book", "past", "adjective"),
    )


def role_options(plan: Plan, role: str, side: str) -> tuple[Realization, ...]:
    # Alternatives are independent realizations of the same referent/event,
    # not mirrored lexical units.  Side-specific forms keep the shared plan
    # semantic while allowing distinct surface choices.
    if role == "SUBJECT":
        if plan.subject_ref == "mara":
            words = ("Mara", "the reader") if side == "left" else ("Mara", "the scribe")
        elif plan.subject_ref == "noah":
            words = ("Noah", "the sailor") if side == "left" else ("Noah", "the keeper")
        else:
            words = ("the keeper", "the guard")
        return tuple(Realization(role, w, plan.subject_ref, "subject") for w in words)
    if role == "VERB":
        table = {
            "read": ("reads", "studies", "reads") if plan.tense == "present" else ("read", "studied", "read"),
            "open": ("opens", "unbars", "opens"),
            "guard": ("guards", "keeps", "guarded"),
        }
        words = table[plan.event_ref]
        if side == "right": words = words[1:]
        return tuple(Realization(role, w, plan.event_ref, plan.tense) for w in words)
    if role == "OBJECT":
        table = {
            "letter": ("the letter", "a letter", "the message"),
            "gate": ("the gate", "a gate", "the door"),
            "book": ("the book", "a book", "the ledger"),
        }
        return tuple(Realization(role, w, plan.object_ref, "object") for w in table[plan.object_ref])
    if role == "ATTACHMENT":
        table = {
            "pp": ("at dawn", "by the fire", "in the hall"),
            "adverb": ("quietly", "at once", "softly"),
            "adjective": ("in the old room", "under the high roof", "near the gate"),
        }
        return tuple(Realization(role, w, plan.attachment, "attachment") for w in table[plan.attachment])
    return ()


def prose_controls() -> list[dict[str, object]]:
    texts = [
        "Mara reads the letter at dawn.", "Mara reads a letter by the fire.",
        "Mara studies the message in the hall.", "The reader reads the letter quietly.",
        "The reader studies a letter at once.", "Noah opens the gate quietly.",
        "Noah opens a gate at dawn.", "The sailor unbars the door by the fire.",
        "The sailor opens the gate softly.", "Noah opens the door in the hall.",
        "The keeper guarded the book in the old room.", "The keeper guarded a book under the high roof.",
        "The guard kept the ledger near the gate.", "The keeper guarded the book at dawn.",
        "The guard kept a book in the hall.", "Mara reads the message softly.",
        "Noah opens the door at once.", "The reader studies the letter by the fire.",
        "The guard kept the book under the high roof.", "The keeper guarded the ledger near the gate.",
    ]
    return [{"rendered": text, "audit": audit(text),
             "reader_status": "complete contemporary prose control; not exact"}
            for text in texts]


def run(*, state_limit: int = 200_000) -> dict[str, object]:
    states = pruned = advances = stack_pruned = 0
    candidates: list[dict[str, object]] = []
    witnesses: list[dict[str, object]] = []

    for plan in plans():
        # The open stack is the same semantic clause contract on both sides;
        # each transition pops one constituent before surface characters are
        # admitted to the shared equation.
        initial_stack = ("SUBJECT", "VERB", "OBJECT", "ATTACHMENT")

        def walk(stack_left: tuple[str, ...], stack_right: tuple[str, ...],
                 left: str, right: str,
                 ls: tuple[Realization, ...], rs: tuple[Realization, ...]) -> None:
            nonlocal states, pruned, advances, stack_pruned
            if states >= state_limit:
                return
            if not stack_left and not stack_right:
                if left or right:
                    return
                ordered = ls + tuple(reversed(rs))
                rendered = " ".join(x.text for x in ordered) + "."
                checked = audit(rendered)
                if checked["exact"]:
                    candidates.append({"rendered": rendered, "audit": checked,
                        "provenance": {"construction": "discourse-plan-conditioned delayed realization",
                            "plan": plan.plan_id, "shared_referents": [plan.subject_ref, plan.event_ref, plan.object_ref],
                            "tense": plan.tense, "attachment": plan.attachment,
                            "open_constituent_stack": True, "stack_transitions": len(ls) + len(rs),
                            "finished_tape_reversal": False, "post_hoc_repair": False,
                            "catalogue_text": False, "aligned_token_mirror": False},
                        "reader_status": "unreviewed; exactness does not certify readability"})
                return
            if not stack_left or not stack_right:
                stack_pruned += 1
                return
            role_left, role_right = stack_left[0], stack_right[-1]
            if role_left != role_right:
                stack_pruned += 1
                return
            for lreal in role_options(plan, role_left, "left"):
                for rreal in role_options(plan, role_right, "right"):
                    states += 1
                    nl = left + letters(lreal.text)
                    nr = letters(rreal.text) + right
                    residual = consume(nl, nr)
                    if residual is None:
                        pruned += 1
                        if len(witnesses) < 20:
                            z = " ".join(x.text for x in ls + (lreal,) + (rreal,) + tuple(reversed(rs))) + "."
                            witnesses.append({"rendered": z, "depth": len(ls), "audit": audit(z),
                                              "reader_status": "diagnostic witness; not a candidate"})
                        continue
                    advances += 1
                    walk(stack_left[1:], stack_right[:-1], residual[0], residual[1],
                         ls + (lreal,), (rreal,) + rs)

        walk(initial_stack, tuple(reversed(initial_stack)), "", "", (), ())

    candidates.sort(key=lambda row: row["audit"]["letters"], reverse=True)
    result = {"experiment": EXPERIMENT_ID,
              "method": "synchronous discourse-conditioned open-stack grammar",
              "complete_prose_controls": prose_controls(), "candidates": candidates,
              "witnesses": witnesses,
              "stats": {"plans": len(plans()), "states": states, "pruned": pruned,
                        "stack_pruned": stack_pruned, "chart_advances": advances,
                        "exact": len(candidates)},
              "provenance": {"novelty_signature": SIGNATURE,
                  "novelty_preflight": "extended signature is absent from the registry; prior delayed-realization lanes do not expose an open constituent stack with attachment choice before paired output",
                  "shared_referents": True, "explicit_open_constituent_stack": True,
                  "optional_attachment_selected_before_output": True,
                  "independent_pointer_sha_audit": True, "finished_tape_reversal": False,
                  "post_hoc_repair": False, "catalogue_text": False,
                  "aligned_token_mirror": False,
                  "next_construction": "permit two-clause discourse plans with anaphoric pronoun realization while preserving stack ownership"}}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
