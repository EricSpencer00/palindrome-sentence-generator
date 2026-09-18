"""Center-out lexical slot reconciliation on one connected syntax tree.

This is a distinct construction operator from outer-in generation.  A single
feature grammar owns the complete surface tree.  The search exposes the two
leaves adjacent to the tree centre, lexicalizes one, immediately emits its
characters, then lexicalizes the opposite exposed leaf and immediately
reconciles its characters against the active residual.  It continues outward
until every tree leaf is complete.  No left or right sentence is authored
independently, and no partial surface is admitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS, MAX_LETTERS = 100, 180
WORD = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class Lexeme:
    word: str
    category: str
    semantic: str
    subject: str = ""
    object: str = ""


@dataclass(frozen=True)
class Symbol:
    name: str
    role: str = ""
    kind: str = ""
    feature: str = ""


@dataclass(frozen=True)
class Node:
    symbol: Symbol
    children: tuple["Node", ...] = ()


@dataclass(frozen=True)
class Slot:
    index: int
    symbol: Symbol


LEXICON = (
    Lexeme("a", "det", "determiner"), Lexeme("an", "det", "determiner"),
    Lexeme("the", "det", "determiner"), Lexeme("this", "det", "determiner"),
    Lexeme("that", "det", "determiner"), Lexeme("because", "conn", "causal"),
    Lexeme("after", "prep", "temporal"), Lexeme("today", "adv", "temporal"),
    Lexeme("expert", "adj_person", "person"), Lexeme("patient", "adj_person", "person"),
    Lexeme("careful", "adj_person", "person"), Lexeme("calm", "adj_person", "person"),
    Lexeme("detailed", "adj_artifact", "artifact"), Lexeme("damaged", "adj_artifact", "artifact"),
    Lexeme("rare", "adj_artifact", "artifact"), Lexeme("valuable", "adj_artifact", "artifact"),
    Lexeme("annual", "adj_event", "event"), Lexeme("public", "adj_event", "event"),
    Lexeme("archivist", "noun_person", "person"), Lexeme("curator", "noun_person", "person"),
    Lexeme("editor", "noun_person", "person"), Lexeme("researcher", "noun_person", "person"),
    Lexeme("artist", "noun_person", "person"), Lexeme("manuscript", "noun_artifact", "artifact"),
    Lexeme("artifact", "noun_artifact", "artifact"), Lexeme("collection", "noun_artifact", "artifact"),
    Lexeme("record", "noun_artifact", "artifact"), Lexeme("report", "noun_artifact", "artifact"),
    Lexeme("study", "noun_event", "event"), Lexeme("trial", "noun_event", "event"),
    Lexeme("review", "noun_event", "event"),
    Lexeme("reviews", "verb", "review", "person", "artifact"),
    Lexeme("records", "verb", "record", "person", "artifact"),
    Lexeme("repairs", "verb", "repair", "person", "artifact"),
    Lexeme("catalogs", "verb", "record", "person", "artifact"),
)
BY_CATEGORY: dict[str, tuple[Lexeme, ...]] = {}
for _item in LEXICON:
    BY_CATEGORY.setdefault(_item.category, ())
    BY_CATEGORY[_item.category] += (_item,)


def terminal(kind: str, role: str) -> Symbol:
    return Symbol("T", role=role, kind=kind)


class FeatureGrammar:
    """A single causal tree; every token is a terminal descendant of S."""

    def start(self) -> Symbol:
        return Symbol("S")

    def productions(self, symbol: Symbol) -> tuple[tuple[Symbol, ...], ...]:
        if symbol.name == "S":
            return ((Symbol("NP", role="subject", kind="person"),
                     Symbol("VP", role="matrix"), Symbol("PP", role="temporal"),
                     terminal("conn", "because"), Symbol("CLAUSE", role="cause"),
                     terminal("adv", "today")),)
        if symbol.name == "VP":
            return ((terminal("verb", f"{symbol.role}_verb"),
                     Symbol("NP", role=f"{symbol.role}_object", kind="artifact")),)
        if symbol.name == "CLAUSE":
            return ((Symbol("NP", role="cause_subject", kind="person"),
                     terminal("verb", "cause_verb"),
                     Symbol("NP", role="cause_object", kind="artifact")),)
        if symbol.name == "PP":
            return ((terminal("prep", "after"), Symbol("NP", role="event", kind="event")),)
        if symbol.name == "NP":
            return ((terminal("det", f"{symbol.role}_det"),
                     terminal(f"adj_{symbol.kind}", f"{symbol.role}_adj"),
                     terminal(f"noun_{symbol.kind}", symbol.role)),)
        return ()

    def expand(self, symbol: Symbol) -> Node:
        rhs = self.productions(symbol)
        return Node(symbol, tuple(self.expand(child) for child in rhs[0])) if rhs else Node(symbol)

    def digest(self) -> str:
        seen: set[Symbol] = set(); queue = [self.start()]; rows = []
        while queue:
            symbol = queue.pop(0)
            if symbol in seen:
                continue
            seen.add(symbol)
            rhs = self.productions(symbol)
            rows.append((asdict(symbol), [[asdict(x) for x in row] for row in rhs]))
            queue.extend(x for row in rhs for x in row if x.name != "T")
        return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def slots(tree: Node) -> tuple[Slot, ...]:
    found: list[Slot] = []
    def visit(node: Node) -> None:
        if node.symbol.name == "T":
            found.append(Slot(len(found), node.symbol)); return
        for child in node.children:
            visit(child)
    visit(tree)
    return tuple(found)


def choices(symbol: Symbol) -> tuple[Lexeme, ...]:
    return BY_CATEGORY[symbol.kind]


def determiner_ok(det: str, adjective: str) -> bool:
    initial = adjective[:1].lower()
    return (det == "an" and initial in "aeiou") or (det == "a" and initial not in "aeiou") or det in {"the", "this", "that"}


def semantic_ok(slot: Slot, lexeme: Lexeme, assigned: dict[str, Lexeme]) -> bool:
    if lexeme.word in {x.word for x in assigned.values()}:
        return False
    if slot.symbol.kind == "verb":
        if (lexeme.subject, lexeme.object) != ("person", "artifact"):
            return False
    return True


def render(words: Iterable[str]) -> str:
    text = " ".join(words)
    return text[:1].upper() + text[1:] + "."


def emit_left_outward(left_residual: str, right_residual: str, word: str, stats: Counter) -> tuple[str, str] | None:
    """Emit a left leaf from its inner edge (right to left)."""
    for char in reversed(normalize_letters(word)):
        stats["left_char_emissions"] += 1
        if right_residual:
            if char != right_residual[0]:
                stats["residual_contradictions"] += 1; return None
            right_residual = right_residual[1:]
            stats["residual_cancellations"] += 1
        else:
            left_residual += char
    return left_residual, right_residual


def emit_right_outward(left_residual: str, right_residual: str, word: str, stats: Counter) -> tuple[str, str] | None:
    """Emit a right leaf from its inner edge (left to right)."""
    for char in normalize_letters(word):
        stats["right_char_emissions"] += 1
        if left_residual:
            if char != left_residual[0]:
                stats["residual_contradictions"] += 1; return None
            left_residual = left_residual[1:]
            stats["residual_cancellations"] += 1
        else:
            right_residual += char
    return left_residual, right_residual


def exact_audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    mismatches = [(i, len(tape) - 1 - i) for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]]
    cursor, bounds = 0, []
    for token in WORD.findall(text.lower()):
        cursor += len(token); bounds.append(cursor)
    return {"exact": bool(tape) and not mismatches, "letters": len(tape), "mismatches": mismatches,
            "normalized_sha256": sha256(tape.encode()).hexdigest(),
            "shifted_word_boundaries": sorted(set(bounds[:-1]) - {len(tape) - x for x in bounds[:-1]})}


def parse_tree(grammar: FeatureGrammar, text: str) -> Node | None:
    if text != text.strip() or re.sub(r"[A-Za-z ,.?!'-]", "", text):
        return None
    tokens = tuple(WORD.findall(text.lower()))
    tree = grammar.expand(grammar.start()); expected = slots(tree)
    if len(tokens) != len(expected):
        return None
    for token, slot in zip(tokens, expected):
        if not any(token == item.word for item in choices(slot.symbol)):
            return None
    for i, slot in enumerate(expected):
        if slot.symbol.kind == "det" and not determiner_ok(tokens[i], tokens[i + 1]):
            return None
    by_role = {slot.symbol.role: tokens[i] for i, slot in enumerate(expected)}
    verbs = {item.word for item in BY_CATEGORY["verb"] if item.subject == "person" and item.object == "artifact"}
    if by_role.get("matrix_verb") not in verbs or by_role.get("cause_verb") not in verbs:
        return None
    return tree


def audit(grammar: FeatureGrammar, text: str, kind: str, provenance: tuple[str, ...]) -> dict[str, object]:
    exact = exact_audit(text); parsed = parse_tree(grammar, text)
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [key for key, value in central.items() if not value]
    if parsed is None:
        codes.append("independent_complete_reparse_failed")
    return {"record_kind": kind, "rendered": text, "provenance": provenance,
            "independent_exact_audit": exact, "independent_parse": parsed is not None,
            "central_admission": central, "mechanically_admitted": not codes,
            "rejection_codes": codes, "reader_status": "unreviewed; programmatic checks do not certify readability"}


def center_out_search(leaf_slots: tuple[Slot, ...], grammar: FeatureGrammar, *, state_limit: int, closure_limit: int, stats: Counter) -> list[dict[str, object]]:
    if len(leaf_slots) % 2:
        raise ValueError("center-out operator requires an even leaf frontier")
    rows: list[dict[str, object]] = []; words = [""] * len(leaf_slots); center = len(leaf_slots) // 2

    def visit(lo: int, hi: int, assigned: dict[str, Lexeme], left_residual: str, right_residual: str, owner: int, last_side: int) -> None:
        if len(rows) >= closure_limit or stats["state_count"] >= state_limit:
            if stats["state_count"] >= state_limit: stats["state_limit_reached"] += 1
            return
        if lo < 0 and hi >= len(leaf_slots):
            stats["complete_tree_states"] += 1
            residual = left_residual or right_residual
            if residual and residual != residual[::-1]:
                stats["central_residual_rejections"] += 1; return
            text = render(words); row = audit(grammar, text, "complete_center_out_closure", tuple(words))
            if row["mechanically_admitted"]: rows.append(row)
            else: stats["mechanical_rejections"] += 1
            return
        if left_residual:
            side_options = (-1,) if hi < len(leaf_slots) else ()
        elif right_residual:
            side_options = (1,) if lo >= 0 else ()
        else:
            side_options = (1 if last_side in {0, -1} else -1,)
            if side_options[0] == 1 and lo < 0: side_options = (-1,)
            if side_options[0] == -1 and hi >= len(leaf_slots): side_options = (1,)
        if not side_options:
            stats["unmatched_frontier_rejections"] += 1; return
        side = side_options[0]; index = lo if side == 1 else hi; slot = leaf_slots[index]
        for item in choices(slot.symbol):
            if not semantic_ok(slot, item, assigned):
                continue
            stats["state_count"] += 1
            if side == 1:
                updated = emit_left_outward(left_residual, right_residual, item.word, stats)
            else:
                updated = emit_right_outward(left_residual, right_residual, item.word, stats)
            if updated is None:
                continue
            assigned[slot.symbol.role] = item; words[index] = item.word
            if side == 1: stats["left_leaf_emissions"] += 1
            else: stats["right_leaf_emissions"] += 1
            visit(lo - (1 if side == 1 else 0), hi + (1 if side == -1 else 0), assigned,
                  updated[0], updated[1], 1 if updated[0] else (-1 if updated[1] else 0), side)
            words[index] = ""; assigned.pop(slot.symbol.role, None)

    visit(center - 1, center, {}, "", "", 0, 0)
    return rows


def center_boundary_feasibility(leaf_slots: tuple[Slot, ...]) -> dict[str, object]:
    """Report whether the first exposed centre slots share any initial letter."""
    center = len(leaf_slots) // 2
    left, right = leaf_slots[center - 1], leaf_slots[center]
    compatible = []
    for l_item in choices(left.symbol):
        for r_item in choices(right.symbol):
            if normalize_letters(l_item.word)[-1] == normalize_letters(r_item.word)[0]:
                compatible.append((l_item.word, r_item.word))
    return {"left_role": left.symbol.role, "left_kind": left.symbol.kind,
            "right_role": right.symbol.role, "right_kind": right.symbol.kind,
            "compatible_first_character_pairs": compatible,
            "compatible_pair_count": len(compatible)}


def run(*, state_limit: int = 100_000, closure_limit: int = 100) -> dict[str, object]:
    grammar = FeatureGrammar(); tree = grammar.expand(grammar.start()); leaf_slots = slots(tree)
    stats = Counter(state_count=0, left_char_emissions=0, right_char_emissions=0,
                    left_leaf_emissions=0, right_leaf_emissions=0, residual_cancellations=0,
                    residual_contradictions=0, central_residual_rejections=0,
                    complete_tree_states=0, mechanical_rejections=0)
    closures = center_out_search(leaf_slots, grammar, state_limit=state_limit, closure_limit=closure_limit, stats=stats)
    controls = (
        "An expert archivist reviews a detailed manuscript after the annual study because a patient curator records the damaged artifact today.",
        "A patient researcher catalogs a valuable manuscript after the public trial because an expert archivist repairs the detailed collection today.",
    )
    control_rows = [audit(grammar, text, "complete_connected_grammar_control", tuple(WORD.findall(text.lower()))) for text in controls]
    return {"status": "center_out_lexical_slot_reconciliation_single_tree", "config": {
        "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "state_limit": state_limit,
        "closure_limit": closure_limit, "one_connected_tree": True, "grammar_owns_every_leaf": True,
        "center_out_exposed_slot_schedule": True, "character_residual_per_emission": True,
        "closure_requires_complete_tree": True, "independent_complete_reparse": True,
        "corpus_or_catalogue_generation": False},
        "center_boundary_feasibility": center_boundary_feasibility(leaf_slots),
        "stats": dict(stats), "exact_closures": closures,
        "complete_grammar_controls": control_rows,
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "grammar_sha256": grammar.digest(), "material": "task-authored connected feature grammar and lexicon; no corpus/catalogue text"},
        "reader_facing_next_operator": "Add a typed center production with two semantically compatible noun leaves, preserving this outward residual schedule and complete-tree reparse gate.",
        "scope": "Exactness and feature parsing are mechanical filters; readability requires blinded human evidence."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-limit", type=int, default=100_000); parser.add_argument("--closure-limit", type=int, default=100)
    args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit, closure_limit=args.closure_limit); args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["state_count"], "closures": len(result["exact_closures"]), "controls": len(result["complete_grammar_controls"])}, indent=2))


if __name__ == "__main__":
    main()
