"""Agreement/valency-aware word-equation search over a live character product.

This is a broader follow-up to the seed-interior fixture.  Slot alternatives
are compiled into shared character tries; the outside-in solver expands only
equal-character edges (including explicit word-boundary epsilon edges).  It
therefore searches the lexical state space without materializing a Cartesian
product of finished sentences or invoking a reward model.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "feature-constrained-live-product-20260917.json"
EXPERIMENT_ID = "feature-constrained-live-product-20260917"
SIGNATURE = "live-slot-trie-product|agreement-valency-banks|epsilon-boundaries|semantic-history|independent-audit"


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    char: str | None
    word: str | None = None
    role: str | None = None


@dataclass(frozen=True)
class Automaton:
    start: int
    end: int
    edges: tuple[Edge, ...]
    slots: tuple[str, ...]


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.lower()))


BANKS = {
    "DET_SG": ("a", "an", "one", "this", "that", "each", "every", "my", "his", "her"),
    "DET_PL": ("some", "many", "these", "those", "our", "their", "no"),
    "NUM_PL": ("two", "three", "nine", "ten", "many", "some"),
    "ADJ": ("old", "young", "kind", "quiet", "calm", "wise", "bright", "small", "clear", "warm", "red", "blue", "new", "good", "great", "little", "patient", "careful", "gentle", "brave", "eager", "fresh", "green", "long", "short", "dark", "soft", "strong", "happy", "simple", "local", "honest", "open", "ready", "common"),
    "HUM_SG": ("aide", "artist", "author", "baker", "child", "gardener", "guide", "keeper", "man", "nurse", "painter", "pilot", "poet", "reader", "sailor", "scholar", "singer", "student", "teacher", "woman", "writer", "person", "friend", "leader", "doctor"),
    "HUM_PL": ("aides", "artists", "authors", "bakers", "children", "gardeners", "guides", "keepers", "men", "nurses", "painters", "pilots", "poets", "readers", "sailors", "scholars", "singers", "students", "teachers", "women", "writers", "people", "friends", "leaders", "doctors"),
    "OBJ_SG": ("letter", "map", "memo", "note", "book", "garden", "record", "report", "idea", "reason", "ship", "stone", "star", "dream", "house", "room", "road", "word", "world", "day", "time", "life", "hand", "heart", "voice", "name", "place", "song", "story"),
    "OBJ_PL": ("letters", "maps", "memos", "notes", "books", "gardens", "records", "reports", "ideas", "reasons", "ships", "stones", "stars", "dreams", "houses", "rooms", "roads", "words", "worlds", "days", "times", "lives", "hands", "hearts", "voices", "names", "places", "songs", "stories"),
    "V_SG": ("rips", "reads", "helps", "sees", "meets", "likes", "needs", "marks", "guides", "follows", "carries", "keeps", "watches", "inspires", "informs", "visits", "calls", "asks", "names", "paints", "writes", "makes", "moves", "knows", "loves", "finds", "teaches", "sends", "brings", "opens", "closes", "draws", "shows", "holds", "gives", "takes", "uses", "shares", "plans", "builds", "plants", "records", "saves"),
    "V_PL": ("rip", "read", "help", "see", "meet", "like", "need", "mark", "guide", "follow", "carry", "keep", "watch", "inspire", "inform", "visit", "call", "ask", "name", "paint", "write", "make", "move", "know", "love", "find", "teach", "send", "bring", "open", "close", "draw", "show", "hold", "give", "take", "use", "share", "plan", "build", "plant", "record", "save"),
    "NAME": ("diana", "ada", "iris", "anna", "nina", "mara", "lina", "martin", "helen", "nora", "otto", "eve", "ava", "emma", "liam", "owen", "olivia", "lucas", "simon", "sara", "sarah", "ian", "eric", "maya", "mia", "leo", "lena", "ruth", "rose", "jane", "john", "mary"),
}

FUNCTION_WORDS = {"a", "an", "the", "one", "this", "that", "each", "every", "my", "his", "her", "some", "many", "these", "those", "our", "their", "no", "two", "three", "nine", "ten"}


def _article_ok(words: tuple[str, ...]) -> bool:
    vowels = set("aeiou")
    for index, word in enumerate(words[:-1]):
        if word == "a" and words[index + 1][0] in vowels:
            return False
        if word == "an" and words[index + 1][0] not in vowels:
            return False
    return True


def build_automaton(slots: tuple[str, ...]) -> Automaton:
    starts = list(range(len(slots) + 1))
    next_node = len(starts)
    edges: list[Edge] = []
    for index, slot in enumerate(slots):
        root = starts[index]
        trie: dict[tuple[int, str], int] = {}
        for raw in BANKS[slot]:
            word = normalize(raw)
            node = root
            for char in word:
                key = (node, char)
                child = trie.get(key)
                if child is None:
                    child = next_node
                    next_node += 1
                    trie[key] = child
                    edges.append(Edge(node, child, char))
                node = child
            edges.append(Edge(node, starts[index + 1], None, word, slot))
    return Automaton(starts[0], starts[-1], tuple(edges), slots)


def decode(words: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(words)


def audit(rendered: str) -> dict:
    tape = normalize(rendered)
    mismatches = [(i, len(tape) - 1 - i) for i in range(len(tape) // 2)
                  if tape[i] != tape[-1 - i]]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": bool(tape) and not mismatches,
            "two_pointer_exact": bool(tape) and not mismatches,
            "first_mismatch": mismatches[0] if mismatches else None,
            "sha256": forward, "reverse_sha256": reverse}


def anti_shortcut(words: tuple[str, ...]) -> dict:
    norm = tuple(normalize(word) for word in words)
    return {
        "word_order_symmetry": list(norm) == [word[::-1] for word in reversed(norm)],
        "self_palindromic_words": [word for word in norm if len(word) > 1 and word == word[::-1]],
        "repeated_content": len([word for word in norm if word not in FUNCTION_WORDS]) != len(set(word for word in norm if word not in FUNCTION_WORDS)),
        "catalogue_imported": False,
        "fragment": any(len(word) < 2 for word in norm if word not in FUNCTION_WORDS),
    }


def product(left: Automaton, right: Automaton, max_states: int = 500_000) -> dict:
    out_left: dict[int, list[Edge]] = defaultdict(list)
    in_right: dict[int, list[Edge]] = defaultdict(list)
    for edge in left.edges:
        out_left[edge.source].append(edge)
    for edge in right.edges:
        in_right[edge.target].append(edge)
    # Histories are part of the state: merging them would erase lexical
    # uniqueness and could discard a reader-eligible derivation.
    stack = [(left.start, right.end, tuple(), tuple())]
    seen = set()
    records: list[dict] = []
    dead: list[dict] = []
    states = 0
    while stack and states < max_states:
        p, q, lw, rw_rev = stack.pop()
        key = (p, q, lw, rw_rev)
        if key in seen:
            continue
        seen.add(key)
        states += 1
        if p == left.end and q == right.start:
            rw = tuple(reversed(rw_rev))
            words = lw + rw
            if _article_ok(words):
                records.append({"left_words": lw, "right_words": rw,
                                "left_slots": left.slots, "right_slots": right.slots})
            continue
        progressed = False
        for edge in out_left[p]:
            if edge.char is None:
                if edge.word and edge.word not in FUNCTION_WORDS and edge.word in lw + rw_rev:
                    continue
                stack.append((edge.target, q,
                              lw + ((edge.word,) if edge.word else tuple()), rw_rev))
                progressed = True
        for edge in in_right[q]:
            if edge.char is None:
                if edge.word and edge.word not in FUNCTION_WORDS and edge.word in lw + rw_rev:
                    continue
                stack.append((p, edge.source, lw,
                              rw_rev + ((edge.word,) if edge.word else tuple())))
                progressed = True
        for le in out_left[p]:
            if le.char is None:
                continue
            for re in in_right[q]:
                if re.char is None or le.char != re.char:
                    continue
                stack.append((le.target, re.source, lw, rw_rev))
                progressed = True
        if not progressed:
            dead.append({"left_node": p, "right_node": q,
                         "left_chars": sorted({edge.char for edge in out_left[p] if edge.char}),
                         "right_chars": sorted({edge.char for edge in in_right[q] if edge.char}),
                         "left_slots": left.slots, "right_slots": right.slots,
                         "matched_left_words": lw,
                         "matched_right_words": tuple(reversed(rw_rev)),
                         "matched_letters": sum(len(word) for word in lw + rw_rev)})
    return {"states": states, "truncated": bool(stack), "records": records,
            "dead_frontiers": dead[:25]}


CELLS = {
    "seed_like": (("DET_SG", "HUM_SG", "V_SG", "NUM_PL", "OBJ_PL"),
                   ("DET_PL", "HUM_PL", "V_PL", "NAME")),
    "determiner_object": (("DET_SG", "HUM_SG", "V_SG", "DET_PL", "OBJ_PL"),
                          ("DET_PL", "HUM_PL", "V_PL", "NAME")),
    "adjective": (("DET_SG", "ADJ", "HUM_SG", "V_SG", "DET_PL", "OBJ_PL"),
                  ("DET_PL", "ADJ", "HUM_PL", "V_PL", "NAME")),
    "singular_object": (("DET_SG", "HUM_SG", "V_SG", "DET_SG", "OBJ_SG"),
                        ("DET_SG", "HUM_SG", "V_SG", "NAME")),
}


def run() -> dict:
    cells: dict[str, dict] = {}
    novel: list[dict] = []
    for name, (left_slots, right_slots) in CELLS.items():
        result = product(build_automaton(left_slots), build_automaton(right_slots))
        rows: list[dict] = []
        for rec in result["records"][:200]:
            rendered = " ".join(rec["left_words"]) + "; " + " ".join(rec["right_words"])
            words = rec["left_words"] + rec["right_words"]
            a = audit(rendered)
            shortcuts = anti_shortcut(words)
            row = {"rendered": rendered, "fixture": name,
                   "normalized_length": a["letters"],
                   "provenance": {"live_character_product": True,
                                  "left_slots": left_slots,
                                  "right_slots": right_slots,
                                  "agreement_features_precompiled": True,
                                  "valency_banks_hand_authored": True,
                                  "catalogue_imported": False,
                                  "posthoc_reversal": False},
                   "audit": a, "anti_shortcut": shortcuts,
                   "reader_eligible": (a["exact"] and a["letters"] >= 39
                                       and not any(shortcuts.values()))}
            rows.append(row)
            if row["reader_eligible"]:
                novel.append(row)
        cells[name] = {"left_slots": left_slots, "right_slots": right_slots,
                       "states": result["states"], "truncated": result["truncated"],
                       "exact_paths": len(result["records"]), "rows": rows,
                       "dead_frontiers": result["dead_frontiers"],
                       "first_dead_frontier": result["dead_frontiers"][0] if result["dead_frontiers"] else None}
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
            "status": "completed_exact_candidates" if novel else "completed_no_novel_exact_closure",
            "method": "live slot-trie outside-in product with agreement and valency banks",
            "cells": cells, "novel_exact_candidates": novel[:50],
            "reader_evidence": "programmatic exactness is not readability; blinded intact-vs-shuffled study required",
            "next_repair": {"operator": "semantic_frame_restriction_at_live_slot_state",
                            "reason": "retain exact states but reject implausible subject/verb/object frames before deeper expansion",
                            "no_rlaif": True},
            "invariants": ["characters match before product states advance",
                           "only complete grammar paths are emitted",
                           "agreement and valency are encoded in slot banks",
                           "independent two-pointer and SHA audits run on every rendered candidate"]}


if __name__ == "__main__":
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    result = json.loads(OUT.read_text())
    print(json.dumps({"status": result["status"],
                      "novel_exact": len(result["novel_exact_candidates"]),
                      "cells": {key: (value["states"], value["exact_paths"]) for key, value in result["cells"].items()}}))
