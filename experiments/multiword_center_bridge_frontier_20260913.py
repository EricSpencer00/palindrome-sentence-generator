"""A live character bridge owned by one connected sentence grammar.

The centre is the ordinary infinitival phrase ``to order red root``.  Its
normalized letters are ``toorderredroot``, which is a palindrome because the
word boundary crosses the character seam: ``toorder`` / ``redroot``.  No
individual word is the reverse of another.  The bridge is traced one emitted
character at a time before the surrounding tree is exposed.

This is a construction experiment, not readability evidence.  A complete
surface is independently reparsed and sent through the central admission gate
before it can be counted as a closure.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS, MAX_LETTERS = 100, 220
WORD = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class Lexeme:
    word: str
    kind: str
    meaning: str
    subject_type: str = ""
    object_type: str = ""


@dataclass(frozen=True)
class Symbol:
    name: str
    role: str = ""
    kind: str = ""


@dataclass(frozen=True)
class Node:
    symbol: Symbol
    children: tuple["Node", ...] = ()


@dataclass(frozen=True)
class Slot:
    index: int
    symbol: Symbol


def T(kind: str, role: str) -> Symbol:
    return Symbol("T", role=role, kind=kind)


LEXICON = (
    Lexeme("a", "det", "indefinite"), Lexeme("an", "det", "indefinite"),
    Lexeme("the", "det", "definite"), Lexeme("this", "det", "definite"),
    Lexeme("that", "det", "definite"),
    Lexeme("during", "prep", "temporal"), Lexeme("after", "prep", "temporal"),
    Lexeme("for", "prep", "benefactive"), Lexeme("near", "prep", "locative"),
    Lexeme("with", "prep", "accompaniment"), Lexeme("in", "prep", "locative"),
    Lexeme("long", "adj_event", "duration"), Lexeme("detailed", "adj_event", "event"),
    Lexeme("careful", "adj_person", "person"), Lexeme("young", "adj_person", "person"),
    Lexeme("patient", "adj_person", "person"), Lexeme("diligent", "adj_person", "person"),
    Lexeme("red", "adj_artifact", "color"),
    Lexeme("trial", "noun_event", "event"), Lexeme("study", "noun_event", "event"),
    Lexeme("chef", "noun_person", "person"), Lexeme("editor", "noun_person", "person"),
    Lexeme("researcher", "noun_person", "person"), Lexeme("museum", "noun_org", "organization"),
    Lexeme("archive", "noun_org", "organization"), Lexeme("root", "noun_artifact", "food"),
    Lexeme("decided", "verb", "decision", "person", "event"),
    Lexeme("deliberately", "adv", "manner"), Lexeme("today", "adv", "temporal"),
    Lexeme("to", "inf", "infinitive"), Lexeme("order", "verb", "ordering", "person", "food"),
)
BY_KIND: dict[str, tuple[Lexeme, ...]] = {}
for item in LEXICON:
    BY_KIND.setdefault(item.kind, ())
    BY_KIND[item.kind] += (item,)


class Grammar:
    """One connected S tree; the bridge is a child node, not a second text."""

    def start(self) -> Symbol:
        return Symbol("S")

    def productions(self, s: Symbol) -> tuple[tuple[Symbol, ...], ...]:
        if s.name == "S":
            return ((Symbol("PP", "during", "event_pp"), Symbol("PP", "after", "event_pp"),
                     Symbol("NP", "subject", "person"), T("adv", "manner"),
                     T("verb", "matrix_verb"), Symbol("BRIDGE", "center", "ordered_food"),
                     Symbol("PP", "for", "benefactive"), Symbol("PP", "near", "locative"),
                     Symbol("PP", "with", "accompaniment"), Symbol("PP", "in", "locative")),)
        if s.name == "BRIDGE":
            return ((T("inf", "bridge_inf"), T("verb", "bridge_verb"),
                     T("adj_artifact", "bridge_adj"), T("noun_artifact", "bridge_noun")),)
        if s.name == "NP":
            return ((T("det", s.role + "_det"), T("adj_person", s.role + "_adj1"),
                     T("adj_person", s.role + "_adj2"),
                     T("noun_person" if s.kind == "person" else "noun_org", s.role + "_noun")),)
        if s.name == "PP":
            noun_kind = "noun_event" if s.kind == "event_pp" else ("noun_org" if s.role in {"near", "in"} else "noun_person")
            adj_kind = "adj_event" if s.role in {"during", "after"} else "adj_person"
            if s.role in {"near", "in"}:
                return ((T("prep", s.role), T("det", s.role + "_det"), T(noun_kind, s.role + "_noun")),)
            return ((T("prep", s.role), T("det", s.role + "_det"), T(adj_kind, s.role + "_adj"), T(noun_kind, s.role + "_noun")),)
        return ()

    def expand(self, s: Symbol) -> Node:
        rhs = self.productions(s)
        return Node(s, tuple(self.expand(x) for x in rhs[0])) if rhs else Node(s)

    def digest(self) -> str:
        rows, seen, queue = [], set(), [self.start()]
        while queue:
            s = queue.pop(0)
            if s in seen:
                continue
            seen.add(s)
            rhs = self.productions(s)
            rows.append((s.name, s.role, s.kind, [[(x.name, x.role, x.kind) for x in row] for row in rhs]))
            queue.extend(x for row in rhs for x in row if x.name != "T")
        return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def slots(tree: Node) -> tuple[Slot, ...]:
    out: list[Slot] = []
    def visit(node: Node) -> None:
        if node.symbol.name == "T":
            out.append(Slot(len(out), node.symbol))
        else:
            for child in node.children:
                visit(child)
    visit(tree)
    return tuple(out)


def normalize_words(words: Iterable[str]) -> str:
    return normalize_letters(" ".join(words))


BRIDGE_LEFT = ("to", "order")
BRIDGE_RIGHT = ("red", "root")
BRIDGE_LICENSE = {BRIDGE_LEFT + BRIDGE_RIGHT: "an infinitival order of red root food"}


def bridge_trace() -> dict[str, object]:
    """Run the centre scheduler over lexical leaves, one character per step."""
    left, right = normalize_words(BRIDGE_LEFT), normalize_words(BRIDGE_RIGHT)
    trace: list[dict[str, object]] = []
    residual = ""
    cancellations = 0
    # Leaves are exposed near the centre, but a leaf may be paused after one
    # opening character while the opposite frontier cancels it.  Thus no
    # whole word is consumed before the other side participates.
    leaves = {"left": [(w, w[::-1]) for w in (BRIDGE_LEFT[1], BRIDGE_LEFT[0])],
              "right": [(w, w) for w in BRIDGE_RIGHT]}
    cursor = {"left": [0, 0], "right": [0, 0]}
    current = {"left": None, "right": None}
    owner = ""
    preferred = "left"
    while True:
        for side in ("left", "right"):
            if current[side] is not None:
                _, stream = current[side]
                if cursor[side][1] >= len(stream):
                    current[side] = None
        if current["left"] is None and cursor["left"][0] >= len(leaves["left"]):
            left_done = True
        else:
            left_done = False
        if current["right"] is None and cursor["right"][0] >= len(leaves["right"]):
            right_done = True
        else:
            right_done = False
        if left_done and right_done:
            break
        if residual:
            side = "right" if owner == "left" else "left"
        else:
            side = preferred if not (preferred == "left" and left_done) and not (preferred == "right" and right_done) else ("right" if preferred == "left" else "left")
        if current[side] is None:
            if cursor[side][0] >= len(leaves[side]):
                return {"left": left, "right": right, "trace": trace, "completed": False, "cancellations": cancellations}
            leaf_index = cursor[side][0]
            current[side] = leaves[side][leaf_index]
            cursor[side] = [leaf_index + 1, 0]
        word, stream = current[side]
        char = stream[cursor[side][1]]
        cursor[side][1] += 1
        before = residual
        if residual and char != residual[0]:
            trace.append({"side": side, "word": word, "char": char, "residual_before": before, "action": "contradiction"})
            return {"left": left, "right": right, "trace": trace, "completed": False, "cancellations": cancellations}
        if residual:
            residual = residual[1:]
            cancellations += 1
            action = "cancel"
        else:
            residual = char
            owner = side
            action = "open"
        trace.append({"side": side, "word": word, "char": char, "residual_before": before,
                      "residual_after": residual, "action": action})
        if not residual:
            preferred = "left"
    return {"left": left, "right": right, "trace": trace,
            "completed": not residual and len(trace) == len(left) + len(right),
            "cancellations": cancellations}


def exact_audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    mismatches = [(i, len(tape) - 1 - i) for i in range(len(tape) // 2) if tape[i] != tape[-i - 1]]
    return {"exact": bool(tape) and not mismatches, "letters": len(tape),
            "mismatches": mismatches, "normalized_sha256": sha256(tape.encode()).hexdigest()}


def parse_tree(grammar: Grammar, text: str) -> bool:
    if text != text.strip() or re.sub(r"[A-Za-z ,.?!'-]", "", text):
        return False
    tokens = tuple(WORD.findall(text.lower()))
    leaf_slots = slots(grammar.expand(grammar.start()))
    if len(tokens) != len(leaf_slots):
        return False
    assigned: list[Lexeme] = []
    for i, slot in enumerate(leaf_slots):
        options = BY_KIND.get(slot.symbol.kind, ())
        item = next((x for x in options if x.word == tokens[i]), None)
        if item is None:
            return False
        assigned.append(item)
        if item.kind == "det" and item.word in {"a", "an"}:
            if i + 1 >= len(tokens):
                return False
            if (item.word == "an") != (tokens[i + 1][0] in "aeiou"):
                return False
    if tuple(tokens[14:18]) not in BRIDGE_LICENSE:
        return False
    subject = assigned[11]
    matrix = assigned[13]
    bridge_verb = assigned[15]
    if subject.meaning != "person" or matrix.subject_type != "person" or bridge_verb.subject_type != "person":
        return False
    return True


def audit(grammar: Grammar, text: str, kind: str, provenance: tuple[str, ...]) -> dict[str, object]:
    exact = exact_audit(text)
    parsed = parse_tree(grammar, text)
    gate = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [key for key, value in gate.items() if not value]
    if not parsed:
        codes.append("independent_complete_reparse_failed")
    return {"record_kind": kind, "rendered": text, "provenance": provenance,
            "independent_exact_audit": exact, "independent_parse": parsed,
            "central_admission": gate, "mechanically_admitted": not codes,
            "rejection_codes": codes,
            "reader_status": "unreviewed; programmatic checks do not certify readability"}


def outer_trace(words: list[str], leaf_slots: tuple[Slot, ...], center: int, stats: Counter) -> bool:
    """Trace assigned outer leaves in frontier order, never whole-word match."""
    lo, hi, residual, owner = center - 3, center + 4, "", 0
    # center occupies slots center-2..center+3 (four leaves); outer leaves are 0..center-3 and center+4..end.
    while lo >= 0 or hi < len(words):
        side = -owner if residual else (1 if lo >= 0 else -1)
        if side == 1:
            word, stream = words[lo], words[lo][::-1]; lo -= 1
        else:
            word, stream = words[hi], words[hi]; hi += 1
        stats["outer_leaf_exposures"] += 1
        for char in stream:
            stats["outer_char_emissions"] += 1
            if residual:
                if char != residual[0]:
                    stats["outer_residual_contradictions"] += 1
                    return False
                residual = residual[1:]
                stats["outer_cancellations"] += 1
            else:
                residual = char
            owner = side
    return not residual


def search(grammar: Grammar, leaf_slots: tuple[Slot, ...], *, state_limit: int, closure_limit: int, stats: Counter) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    centre = len(leaf_slots) // 2
    bridge = bridge_trace()
    stats["center_char_emissions"] += len(bridge["trace"])
    stats["center_cancellations"] += bridge["cancellations"]
    stats["state_count"] += len(bridge["trace"])
    traces = [bridge]
    if not bridge["completed"]:
        stats["center_bridge_contradictions"] += 1
        return [], traces
    # Expose outer leaves only after the entire bridge is clear.  Each recursive
    # call emits exactly one character; a leaf is lexicalized only when its
    # frontier becomes exposed, and can be paused while the opposite side
    # cancels its residual.
    options = {slot.symbol.kind: BY_KIND.get(slot.symbol.kind, ()) for slot in leaf_slots}
    closures: list[dict[str, object]] = []
    seeded = {i: word for i, word in zip(range(centre - 2, centre + 2), (*BRIDGE_LEFT, *BRIDGE_RIGHT))}

    def visit(lo: int, hi: int, left_word: str, left_pos: int, right_word: str, right_pos: int,
              residual: str, owner: int, preferred: int, assigned: dict[int, str]) -> None:
        if stats["state_count"] >= state_limit or len(closures) >= closure_limit:
            return
        stats["state_count"] += 1
        # A completed leaf is retired before selecting the next exposed leaf.
        if left_word and left_pos >= len(left_word):
            left_word, left_pos = "", 0
        if right_word and right_pos >= len(right_word):
            right_word, right_pos = "", 0
        if lo < 0 and hi >= len(leaf_slots) and not left_word and not right_word:
            if not residual:
                candidate = [assigned[i] for i in range(len(leaf_slots))]
                row = audit(grammar, render(candidate), "connected_frontier_closure", tuple(candidate))
                if row["independent_exact_audit"]["exact"]:
                    stats["exact_closures"] += 1
                if row["mechanically_admitted"]:
                    stats["admitted_closures"] += 1
                closures.append(row)
            return
        # A live residual forces the opposite frontier.  Otherwise continue
        # the preferred frontier if it has an active leaf, then expose a leaf.
        if residual:
            side = -owner
        elif preferred == 1 and (left_word or lo >= 0):
            side = 1
        elif preferred == -1 and (right_word or hi < len(leaf_slots)):
            side = -1
        elif lo >= 0:
            side = 1
        else:
            side = -1
        if side == 1 and not left_word:
            if lo < 0:
                return
            for item in options.get(leaf_slots[lo].symbol.kind, ()):
                if item.word in assigned.values():
                    continue
                stats["outer_leaf_exposures"] += 1
                new_assigned = dict(assigned); new_assigned[lo] = item.word
                visit(lo - 1, hi, item.word[::-1], 0, right_word, right_pos,
                      residual, owner, 1, new_assigned)
            return
        if side == -1 and not right_word:
            if hi >= len(leaf_slots):
                return
            for item in options.get(leaf_slots[hi].symbol.kind, ()):
                if item.word in assigned.values():
                    continue
                stats["outer_leaf_exposures"] += 1
                new_assigned = dict(assigned); new_assigned[hi] = item.word
                visit(lo, hi + 1, left_word, left_pos, item.word, 0,
                      residual, owner, -1, new_assigned)
            return
        word = left_word if side == 1 else right_word
        pos = left_pos if side == 1 else right_pos
        char = word[pos]
        stats["outer_char_emissions"] += 1
        new_residual = residual
        if residual:
            if char != residual[0]:
                stats["outer_residual_contradictions"] += 1
                return
            new_residual = residual[1:]
            stats["outer_cancellations"] += 1
        else:
            new_residual = char
            owner = side
        visit(lo, hi, left_word, left_pos + (side == 1), right_word, right_pos + (side == -1),
              new_residual, owner, preferred, assigned)

    visit(centre - 3, centre + 2, "", 0, "", 0, "", 0, 1, seeded)
    return closures, traces


def render(words: Iterable[str]) -> str:
    text = " ".join(words)
    return text[:1].upper() + text[1:] + "."


def run(*, state_limit: int = 100_000, closure_limit: int = 100) -> dict[str, object]:
    grammar = Grammar()
    leaf_slots = slots(grammar.expand(grammar.start()))
    stats = Counter(state_count=0, center_char_emissions=0, center_cancellations=0,
                    center_bridge_contradictions=0, outer_leaf_exposures=0,
                    outer_char_emissions=0, outer_cancellations=0,
                    outer_residual_contradictions=0, exact_closures=0,
                    admitted_closures=0)
    closures, traces = search(grammar, leaf_slots, state_limit=state_limit, closure_limit=closure_limit, stats=stats)
    controls = (
        "During the long trial after the detailed study the careful young chef deliberately decided to order red root for the patient editor near the archive with a diligent researcher in the museum.",
        "After the long trial during the detailed study the careful young chef deliberately decided to order red root for the patient editor near the archive with a diligent researcher in the museum.",
    )
    rows = [audit(grammar, text, "complete_connected_grammar_control", tuple(WORD.findall(text.lower()))) for text in controls]
    return {"status": "multiword_center_bridge_frontier_single_tree", "config": {
        "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "state_limit": state_limit,
        "closure_limit": closure_limit, "one_connected_tree": True,
        "grammar_owns_every_leaf": True, "center_bridge_before_outer_frontier": True,
        "one_character_emission_states": True, "whole_word_matching": False,
        "semantic_selection_and_agreement_reparse": True, "corpus_or_catalogue_generation": False,
    }, "grammar_leaf_count": len(leaf_slots),
        "center_phrase": {"rendered": "to order red root", "left_words": BRIDGE_LEFT,
                          "right_words": BRIDGE_RIGHT, "meaning": next(iter(BRIDGE_LICENSE.values())),
                          "no_direct_reverse_word_pair": all(a[::-1] != b for a in BRIDGE_LEFT for b in BRIDGE_RIGHT),
                          "normalized": normalize_words(BRIDGE_LEFT + BRIDGE_RIGHT),
                          "normalized_exact": normalize_words(BRIDGE_LEFT + BRIDGE_RIGHT) == normalize_words(BRIDGE_LEFT + BRIDGE_RIGHT)[::-1]},
        "center_bridge_traces": traces, "stats": dict(stats), "exact_closures": closures,
        "complete_grammar_controls": rows,
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "grammar_sha256": grammar.digest(), "material": "task-authored connected grammar and lexical alternatives; no corpus/catalogue text"},
        "reader_facing_next_operator": "After the complete center bridge, add a typed outer-leaf repair that preserves the live residual schedule and retest the full sentence with blinded readers only if an exact admitted closure appears.",
        "scope": "The center is a construction diagnostic; exactness, parsing, and admission do not certify human readability."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-limit", type=int, default=100_000)
    parser.add_argument("--closure-limit", type=int, default=100)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit, closure_limit=args.closure_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["state_count"],
                      "exact": len(result["exact_closures"]),
                      "admitted": result["stats"]["admitted_closures"],
                      "bridge_complete": result["center_bridge_traces"][0]["completed"]}, indent=2))


if __name__ == "__main__":
    main()
