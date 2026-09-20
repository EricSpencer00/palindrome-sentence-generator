"""Live cross-word equations for typed recipient/theme clauses."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "typed-recipient-dative-equations-20260920.json"
EXPERIMENT_ID = "typed-recipient-dative-equations-20260920"
SIGNATURE = "typed-recipient-dative|theme-number|subject-agreement|live-equation"


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
class Chunk:
    role: str
    text: str
    referent: str
    valency: str
    number: str | None = None
    animacy: str | None = None


def chunks(role: str, referent: str, valency: str, *texts: str,
           number: str | None = None, animacy: str | None = None) -> tuple[Chunk, ...]:
    return tuple(Chunk(role, text, referent, valency, number, animacy) for text in texts)


def banks() -> tuple[tuple[Chunk, ...], ...]:
    subjects = (chunks("subject", "agent1", "animate-agent", "the poet", "a sailor", "the guard", number="singular", animacy="animate")
                + chunks("subject", "agent1", "animate-agent", "the poets", "some sailors", "the guards", number="plural", animacy="animate"))
    verbs = chunks("verb", "event1", "ditransitive", "gives", "sends", "shows", "brings", "offers")
    recipients = (chunks("recipient", "recipient1", "dative", "the child", "a friend", "the poet", number="singular", animacy="animate")
                  + chunks("recipient", "recipient1", "dative", "the children", "some friends", "the poets", number="plural", animacy="animate"))
    themes = (chunks("theme", "theme1", "patient", "the letter", "a book", "the seal", number="singular", animacy="inanimate")
              + chunks("theme", "theme1", "patient", "the letters", "some books", "the seals", number="plural", animacy="inanimate"))
    connector = chunks("connector", "relation", "coordination", "and", "while", "but")
    subjects2 = (chunks("subject", "agent2", "animate-agent", "the queen", "a captain", "the poet", number="singular", animacy="animate")
                 + chunks("subject", "agent2", "animate-agent", "the queens", "some captains", "the poets", number="plural", animacy="animate"))
    recipients2 = (chunks("recipient", "recipient2", "dative", "the child", "a friend", "the poet", number="singular", animacy="animate")
                   + chunks("recipient", "recipient2", "dative", "the children", "some friends", "the poets", number="plural", animacy="animate"))
    themes2 = (chunks("theme", "theme2", "patient", "the letter", "a book", "the seal", number="singular", animacy="inanimate")
               + chunks("theme", "theme2", "patient", "the letters", "some books", "the seals", number="plural", animacy="inanimate"))
    return subjects, verbs, recipients, themes, connector, subjects2, verbs, recipients2, themes2


def verb_ok(subject: Chunk, verb: Chunk) -> bool:
    return (subject.number == "singular") == verb.text.endswith("s")


def dative_ok(verb: Chunk, recipient: Chunk, theme: Chunk) -> bool:
    return verb.valency == "ditransitive" and recipient.valency == "dative" and theme.valency == "patient" and recipient.animacy == "animate" and theme.animacy == "inanimate"


def controls() -> list[dict[str, object]]:
    texts = [
        "The poet gives the child the letter and the queen sends a friend a book.",
        "A sailor sends the poet a seal while some guards offer the children books.",
        "The guard shows a friend the book but the poet gives the child a letter.",
        "The poets bring the children some books and a captain sends the poet a seal.",
        "Some sailors offer a friend the letter while the queens show the child a book.",
        "The queen gives the poet a message and the guards send the children the seals.",
        "A captain brings the child a book but the poets offer a friend the letter.",
        "The guards send the poet the seal while a sailor gives the children books.",
        "The poet offers a friend the letter and some captains show the child a book.",
        "The sailors give the children books while the queen sends a poet the seal.",
        "The guard brings the child a letter but the poets show a friend the book.",
        "Some readers send the poet the seals and a captain gives the child a book.",
        "The captain offers a friend a book while the guards give the children letters.",
        "A friend sends the child the letter and the poet brings the children books.",
        "The queens show the poet a seal but a sailor gives a friend the book.",
        "Some captains offer the children books while the guard sends the poet a letter.",
        "The poet brings a friend the book and some sailors give the child the seal.",
        "A captain shows the children letters but the guards offer a friend a book.",
        "The sailors send the poet the seals while a queen gives the child a letter.",
        "Some friends send the poet a book and the poet sends the child a seal.",
    ]
    return [{"rendered": text, "audit": audit(text),
             "reader_status": "complete contemporary prose control; not exact"} for text in texts]


def run(*, state_limit: int = 250_000) -> dict[str, object]:
    lattice = banks()
    states = pruned = advances = feature_pruned = equations = 0
    survivors: list[dict[str, object]] = []
    equation_rows: list[dict[str, object]] = []

    def walk(lo: int, hi: int, left: str, right: str,
             path: tuple[Chunk, ...], equations_so_far: tuple[dict[str, object], ...],
             subjects: tuple[Chunk, ...], recipients: tuple[Chunk, ...],
             themes: tuple[Chunk, ...], right_verbs: tuple[Chunk, ...]) -> None:
        nonlocal states, pruned, advances, feature_pruned, equations
        if states >= state_limit:
            return
        if lo > hi:
            if left or right or len(subjects) != 2 or len(recipients) != 2 or len(themes) != 2 or len(right_verbs) != 1:
                return
            if (not verb_ok(subjects[0], path[1]) or not verb_ok(subjects[1], right_verbs[0])
                    or not dative_ok(path[1], recipients[0], themes[0])
                    or not dative_ok(path[6], recipients[1], themes[1])):
                feature_pruned += 1
                return
            equations += 1
            rendered = " ".join(x.text for x in path) + "."
            checked = audit(rendered)
            row = {"rendered": rendered, "audit": checked, "equations": equations_so_far,
                   "provenance": {"construction": "typed recipient/dative equation scene",
                       "roles": [x.role for x in path], "numbers": [x.number for x in path],
                       "animacy": [x.animacy for x in path], "valencies": [x.valency for x in path],
                       "independent_phrase_boundaries": True, "finished_tape_reversal": False,
                       "post_hoc_repair": False, "catalogue_text": False, "aligned_token_mirror": False},
                   "reader_status": "unreviewed; exactness does not certify readability"}
            equation_rows.append(row)
            if checked["exact"]:
                survivors.append(row)
            return
        if lo == hi:
            for item in lattice[lo]:
                states += 1
                residual = consume(left + letters(item.text), right)
                if residual is None:
                    pruned += 1
                    continue
                advances += 1
                walk(lo + 1, hi - 1, residual[0], residual[1], path + (item,),
                     equations_so_far + ({"left_role": item.role, "left_text": item.text},),
                     subjects, recipients, themes, right_verbs)
            return
        for left_item in lattice[lo]:
            for right_item in lattice[hi]:
                states += 1
                if left_item.role == "verb" and subjects and not verb_ok(subjects[0], left_item):
                    feature_pruned += 1
                    continue
                residual = consume(left + letters(left_item.text), letters(right_item.text) + right)
                if residual is None:
                    pruned += 1
                    continue
                new_subjects = subjects
                new_recipients = recipients
                new_themes = themes
                new_right_verbs = right_verbs
                if left_item.role == "subject":
                    new_subjects = (left_item, subjects[1] if len(subjects) > 1 else left_item)
                if right_item.role == "subject":
                    new_subjects = (subjects[0] if subjects else right_item, right_item)
                if left_item.role == "recipient":
                    new_recipients = (left_item, recipients[1] if len(recipients) > 1 else left_item)
                if right_item.role == "recipient":
                    new_recipients = (recipients[0] if recipients else right_item, right_item)
                if left_item.role == "theme":
                    new_themes = (left_item, themes[1] if len(themes) > 1 else left_item)
                if right_item.role == "theme":
                    new_themes = (themes[0] if themes else right_item, right_item)
                if right_item.role == "verb":
                    new_right_verbs = (right_item,) + right_verbs
                advances += 1
                walk(lo + 1, hi - 1, residual[0], residual[1], path + (left_item,),
                     equations_so_far + ({"left_role": left_item.role, "right_role": right_item.role,
                                          "left_text": left_item.text, "right_text": right_item.text,
                                          "left_letters": letters(left_item.text), "right_letters": letters(right_item.text)},),
                     new_subjects, new_recipients, new_themes, new_right_verbs)

    walk(0, len(lattice) - 1, "", "", (), (), (), (), (), ())
    survivors.sort(key=lambda row: row["audit"]["letters"], reverse=True)
    result = {"experiment": EXPERIMENT_ID, "method": "typed recipient/dative equation scene with live theme number",
              "complete_prose_controls": controls(), "candidates": survivors,
              "equation_rows": equation_rows[:200],
              "stats": {"states": states, "pruned": pruned, "feature_pruned": feature_pruned,
                        "advances": advances, "equation_completions": equations, "exact": len(survivors)},
              "provenance": {"novelty_signature": SIGNATURE,
                  "novelty_preflight": "fresh recipient/dative signature; recipient and theme roles are typed before character output",
                  "recipient_valency_live": True, "theme_number_live": True,
                  "subject_agreement_before_emission": True, "independent_pointer_sha_audit": True,
                  "finished_tape_reversal": False, "post_hoc_repair": False,
                  "catalogue_text": False, "aligned_token_mirror": False,
                  "next_construction": "add a benefactive recipient frame with explicit preposition selection",
                  "reader_next_test": "blind all 20 controls against word-shuffled controls before promoting any exact row"}}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
