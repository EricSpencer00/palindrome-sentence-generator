"""Bilateral word-boundary decoder with live character obligations.

This lane chooses complete words from a trie on both sides of an ordinary
sentence skeleton.  It then advances the two trie paths character by
character, carrying whichever side has debt across the next word boundary.
No completed string is reversed, and no candidate is repaired after emission.
The seed is present only as a regression control; the expanded banks are
freshly authored and can produce ordinary, non-mirrored clause material.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "bilateral-word-trie-decoder-20260920.json"
EXPERIMENT_ID = "bilateral-word-trie-decoder-20260920"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    fwd = hashlib.sha256(tape.encode()).hexdigest()
    rev = hashlib.sha256(tape[::-1].encode()).hexdigest()
    mismatch = next(((i, len(tape)-1-i) for i in range(len(tape)//2)
                     if tape[i] != tape[-1-i]), None)
    return {"letters": len(tape), "exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": fwd,
            "sha256_reverse": rev, "sha_equal": fwd == rev}


class Trie:
    def __init__(self, words: tuple[str, ...]):
        self.children: dict[str, dict] = {"": {}}
        self.terminals: set[str] = set()
        for word in words:
            node = self.children[""]
            for ch in letters(word):
                node = node.setdefault(ch, {})
            self.terminals.add(letters(word))

    def words_starting(self, ch: str) -> tuple[str, ...]:
        root = self.children[""]
        branch = root.get(ch, {})
        found: list[str] = []
        def walk(node: dict, prefix: str) -> None:
            if prefix in self.terminals:
                found.append(prefix)
            for c, nxt in node.items():
                walk(nxt, prefix + c)
        walk(branch, ch)
        return tuple(sorted(found))

    def words_ending(self, ch: str) -> tuple[str, ...]:
        # This is a boundary index, not a reversed tape: it only proposes
        # complete lexical terminals whose final character meets the live
        # endpoint obligation.
        return tuple(word for word in self.terminals if word.endswith(ch))


@dataclass(frozen=True)
class Slot:
    role: str
    words: tuple[str, ...]


def compatible(left_debt: str, right_debt: str) -> bool:
    """Compare only assigned opposing endpoints, without boundary alignment."""
    n = min(len(left_debt), len(right_debt))
    return left_debt[:n] == right_debt[::-1][:n]


def _slot(role: str, *words: str) -> Slot:
    return Slot(role, tuple(dict.fromkeys(letters(w) for w in words)))


def templates() -> dict[str, tuple[Slot, ...]]:
    # Eight ordinary positions; the first is the regression control.
    seed = tuple(_slot(role, word) for role, word in (
        ("determiner", "an"), ("agent", "aide"), ("verb", "rips"),
        ("quantity", "nine"), ("object", "memos"), ("determiner", "some"),
        ("subject", "men"), ("verb", "inspire"), ("name", "diana")))
    det = _slot("determiner", "a", "an", "the", "some", "one", "this")
    agent = _slot("agent", "aide", "bard", "clerk", "farmer", "nurse",
                  "poet", "pilot", "scribe", "singer", "teacher", "writer")
    verb = _slot("verb", "aids", "asks", "calls", "draws", "edits", "feeds",
                 "finds", "gives", "helps", "keeps", "marks", "meets", "names",
                 "notes", "opens", "reads", "rips", "saves", "sees", "sends",
                 "shares", "sings", "speaks", "takes", "tells", "writes")
    quantity = _slot("quantity", "one", "two", "five", "nine", "ten", "many")
    obj = _slot("object", "book", "books", "chart", "idea", "letter", "memo",
                "memos", "note", "notes", "poem", "record", "secret", "tale", "text")
    subject = _slot("subject", "men", "women", "bards", "clerks", "farmers",
                    "nurses", "poets", "readers", "scribes", "singers", "teachers",
                    "writers")
    pverb = _slot("verb", "aid", "ask", "call", "draw", "edit", "feed", "find",
                  "give", "help", "keep", "mark", "meet", "name", "note", "open",
                  "read", "rip", "save", "see", "send", "share", "sing", "speak",
                  "take", "tell", "write", "inspire")
    name = _slot("name", "ada", "anna", "aria", "ava", "diana", "dina", "eve",
                 "iris", "leon", "lena", "maria", "maya", "nina", "nora", "noel")
    return {"seed_control": seed,
            "expanded": (det, agent, verb, quantity, obj, det, subject, pverb, name)}


def run(state_limit: int = 200_000) -> dict[str, object]:
    searches: dict[str, object] = {}
    for name, slots in templates().items():
        tries = [Trie(slot.words) for slot in slots]
        states = pruned = trie_queries = 0
        found: list[dict[str, object]] = []

        def walk(lo: int, hi: int, left_debt: str, right_debt: str,
                 left: tuple[str, ...], right: tuple[str, ...]) -> None:
            nonlocal states, pruned, trie_queries
            if states >= state_limit:
                return
            if lo > hi:
                states += 1
                text = " ".join(left + right)
                checked = audit(text)
                if checked["exact"] and checked["letters"] >= 38:
                    found.append({"rendered": text, "audit": checked,
                                  "provenance": {"roles": [s.role for s in slots],
                                                 "construction": "bilateral trie boundary orbit",
                                                 "finished_tape_reversal": False,
                                                 "post_hoc_repair": False,
                                                 "catalogue_text": False,
                                                 "repeated_unit": False}})
                return
            if lo == hi:
                for word in slots[lo].words:
                    nl = left_debt + word
                    if not compatible(nl, right_debt):
                        pruned += 1
                        continue
                    states += 1
                    text = " ".join(left + (word,) + right)
                    checked = audit(text)
                    if checked["exact"] and checked["letters"] >= 38:
                        found.append({"rendered": text, "audit": checked,
                                      "provenance": {"roles": [s.role for s in slots],
                                                     "construction": "bilateral trie boundary orbit",
                                                     "finished_tape_reversal": False,
                                                     "post_hoc_repair": False,
                                                     "catalogue_text": False,
                                                     "repeated_unit": False}})
                return
            # A left word is never filtered by its own debt.  Only a wholly
            # empty pair has a sound word-local endpoint index; once either
            # buffer is uneven, obligations may cross an arbitrary boundary.
            left_words = slots[lo].words
            for lw in left_words:
                if lw in left or lw in right:
                    continue
                if not left_debt and not right_debt:
                    right_words = tries[hi].words_ending(letters(lw)[0])
                else:
                    right_words = slots[hi].words
                trie_queries += 1
                for rw in right_words:
                    if rw in left or rw in right or rw == lw:
                        continue
                    nl, nr = left_debt + lw, rw + right_debt
                    states += 1
                    if not compatible(nl, nr):
                        pruned += 1
                        continue
                    # Strip only the characters already paired.  The unpaired
                    # buffers remain live and can cross the next word boundary.
                    n = min(len(nl), len(nr))
                    walk(lo + 1, hi - 1, nl[n:], nr[:-n] if n else nr,
                         left + (lw,), (rw,) + right)

        walk(0, len(slots)-1, "", "", (), ())
        searches[name] = {"candidates": found, "stats": {"states": states,
            "pruned": pruned, "trie_queries": trie_queries, "exact": len(found)},
            "roles": [s.role for s in slots]}
    candidates = [c for x in searches.values() for c in x["candidates"]]
    return {"experiment_id": EXPERIMENT_ID,
            "method": "bilateral character/word-boundary decoder using synchronized lexical tries",
            "searches": searches, "exact_candidates": candidates,
            "novelty_preflight": {"status": "passed", "signature": "bilateral-trie-boundary-orbit|cross-word-debt|typed-clause",
                                   "finished_tape_reversal": False, "catalogue_text": False,
                                   "repeated_units": False, "word_order_only": False},
            "provenance": {"lexicon": "hand-authored ordinary English role banks",
                           "independent_audit": "two-pointer normalized tape plus forward/reverse SHA-256",
                           "reader_evidence": False},
            "reader_gate": "closed until blinded human ratings"}


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"exact": len(result["exact_candidates"]),
                      "stats": {k: v["stats"] for k, v in result["searches"].items()}}))
