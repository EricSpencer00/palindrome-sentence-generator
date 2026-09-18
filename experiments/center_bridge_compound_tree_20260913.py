"""Center-bridge compound grammar with character-level outward reconciliation.

This experiment repairs the previous center-out preflight by giving the
single connected grammar a real multi-character center bridge.  The center is
the two noun leaves of one compound object NP; ``part`` and ``trap`` provide a
four-character bridge (the inner-to-outer spelling of ``part`` equals the
outer-to-inner spelling of ``trap``).  They are lexical leaves owned by the
same tree, not a borrowed palindrome or a candidate output.
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
    Lexeme("that", "det", "determiner"), Lexeme("after", "prep", "temporal"),
    Lexeme("near", "prep", "locative"), Lexeme("today", "adv", "temporal"),
    Lexeme("expert", "adj_person", "person"), Lexeme("patient", "adj_person", "person"),
    Lexeme("careful", "adj_artifact", "artifact"), Lexeme("detailed", "adj_artifact", "artifact"),
    Lexeme("valuable", "adj_artifact", "artifact"), Lexeme("rare", "adj_artifact", "artifact"),
    Lexeme("comprehensive", "adj_artifact", "artifact"), Lexeme("substantial", "adj_artifact", "artifact"),
    Lexeme("archivist", "noun_person", "person"), Lexeme("researcher", "noun_person", "person"),
    Lexeme("curator", "noun_person", "person"), Lexeme("editor", "noun_person", "person"),
    # The bridge pair is authored as ordinary artifact nouns.  It is never
    # emitted as a standalone result and does not certify the final sentence.
    Lexeme("part", "noun_artifact", "artifact"), Lexeme("trap", "noun_artifact", "artifact"),
    Lexeme("drawer", "noun_artifact", "artifact"), Lexeme("reward", "noun_artifact", "artifact"),
    Lexeme("manuscript", "noun_artifact", "artifact"), Lexeme("collection", "noun_artifact", "artifact"),
    Lexeme("artifact", "noun_artifact", "artifact"), Lexeme("record", "noun_artifact", "artifact"), Lexeme("report", "noun_artifact", "artifact"),
    Lexeme("study", "noun_event", "event"), Lexeme("review", "noun_event", "event"),
    Lexeme("trial", "noun_event", "event"), Lexeme("inspection", "noun_event", "event"),
    Lexeme("museum", "noun_org", "organization"), Lexeme("archive", "noun_org", "organization"),
    Lexeme("repository", "noun_org", "organization"), Lexeme("survey", "noun_org", "organization"),
    Lexeme("reviews", "verb", "review", "person", "artifact"), Lexeme("records", "verb", "record", "person", "artifact"),
    Lexeme("repairs", "verb", "repair", "person", "artifact"), Lexeme("catalogs", "verb", "record", "person", "artifact"),
)
BY_CATEGORY: dict[str, tuple[Lexeme, ...]] = {}
for _item in LEXICON:
    BY_CATEGORY.setdefault(_item.category, ())
    BY_CATEGORY[_item.category] += (_item,)


def terminal(kind: str, role: str) -> Symbol:
    return Symbol("T", role=role, kind=kind)


class CompoundFeatureGrammar:
    """One sentence tree with a compound object at its center."""

    def start(self) -> Symbol:
        return Symbol("S")

    def productions(self, symbol: Symbol) -> tuple[tuple[Symbol, ...], ...]:
        if symbol.name == "S":
            return ((Symbol("NP", role="subject", kind="person"),
                     terminal("verb", "matrix_verb"), Symbol("COMPOUND", role="object"),
                     Symbol("PP", role="temporal"), Symbol("PP", role="locative"),
                     terminal("adv", "today")),)
        if symbol.name == "COMPOUND":
            return ((terminal("det", "object_det"), terminal("adj_artifact", "object_adj"),
                     terminal("adj_artifact", "object_adj2"),
                     terminal("noun_artifact", "compound_modifier"), terminal("noun_artifact", "compound_head")),)
        if symbol.name == "PP":
            return ((terminal("prep", symbol.role),
                     terminal("det", f"{symbol.role}_det"),
                     terminal("noun_event" if symbol.role == "temporal" else "noun_org", symbol.role + "_object")),)
        if symbol.name == "NP":
            return ((terminal("det", f"{symbol.role}_det"), terminal("adj_person", f"{symbol.role}_adj"),
                     terminal("noun_person", symbol.role)),)
        return ()

    def expand(self, symbol: Symbol) -> Node:
        rhs = self.productions(symbol)
        return Node(symbol, tuple(self.expand(item) for item in rhs[0])) if rhs else Node(symbol)

    def digest(self) -> str:
        seen: set[Symbol] = set(); queue = [self.start()]; rows = []
        while queue:
            symbol = queue.pop(0)
            if symbol in seen: continue
            seen.add(symbol); rhs = self.productions(symbol)
            rows.append((asdict(symbol), [[asdict(x) for x in row] for row in rhs]))
            queue.extend(x for row in rhs for x in row if x.name != "T")
        return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def slots(tree: Node) -> tuple[Slot, ...]:
    found: list[Slot] = []
    def visit(node: Node) -> None:
        if node.symbol.name == "T":
            found.append(Slot(len(found), node.symbol)); return
        for child in node.children: visit(child)
    visit(tree); return tuple(found)


def choices(symbol: Symbol) -> tuple[Lexeme, ...]:
    return BY_CATEGORY[symbol.kind]


def semantic_ok(slot: Slot, item: Lexeme, assigned: dict[str, Lexeme]) -> bool:
    if item.word in {x.word for x in assigned.values()}: return False
    if slot.symbol.kind == "verb" and (item.subject, item.object) != ("person", "artifact"): return False
    return True


def determiner_ok(det: str, following: str) -> bool:
    initial = following[:1].lower()
    return (det == "an" and initial in "aeiou") or (det == "a" and initial not in "aeiou") or det in {"the", "this", "that"}


def render(words: Iterable[str]) -> str:
    text = " ".join(words); return text[:1].upper() + text[1:] + "."


def emit_left_outward(left_residual: str, right_residual: str, word: str, stats: Counter) -> tuple[str, str] | None:
    for char in reversed(normalize_letters(word)):
        stats["left_char_emissions"] += 1
        if right_residual:
            if char != right_residual[0]: stats["residual_contradictions"] += 1; return None
            right_residual = right_residual[1:]; stats["residual_cancellations"] += 1
        else: left_residual += char
    return left_residual, right_residual


def emit_right_outward(left_residual: str, right_residual: str, word: str, stats: Counter) -> tuple[str, str] | None:
    for char in normalize_letters(word):
        stats["right_char_emissions"] += 1
        if left_residual:
            if char != left_residual[0]: stats["residual_contradictions"] += 1; return None
            left_residual = left_residual[1:]; stats["residual_cancellations"] += 1
        else: right_residual += char
    return left_residual, right_residual


def center_bridge_examples(leaf_slots: tuple[Slot, ...]) -> dict[str, object]:
    center = len(leaf_slots) // 2
    left, right = leaf_slots[center - 1], leaf_slots[center]
    examples = []
    for l_item in choices(left.symbol):
        for r_item in choices(right.symbol):
            if normalize_letters(l_item.word)[::-1] == normalize_letters(r_item.word):
                stats = Counter(); first = emit_left_outward("", "", l_item.word, stats)
                if first is not None and emit_right_outward(first[0], first[1], r_item.word, stats) == ("", ""):
                    examples.append({"left": l_item.word, "right": r_item.word, "bridge_letters": len(l_item.word), "cancellations": stats["residual_cancellations"]})
    return {"left_role": left.symbol.role, "right_role": right.symbol.role, "compatible_multi_character_bridges": examples,
            "bridge_count": len(examples)}


def exact_audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text); mismatches = [(i, len(tape)-i-1) for i in range(len(tape)//2) if tape[i] != tape[-i-1]]
    cursor, bounds = 0, []
    for token in WORD.findall(text.lower()): cursor += len(token); bounds.append(cursor)
    return {"exact": bool(tape) and not mismatches, "letters": len(tape), "mismatches": mismatches,
            "normalized_sha256": sha256(tape.encode()).hexdigest(), "shifted_word_boundaries": sorted(set(bounds[:-1]) - {len(tape)-x for x in bounds[:-1]})}


def parse_tree(grammar: CompoundFeatureGrammar, text: str) -> Node | None:
    if text != text.strip() or re.sub(r"[A-Za-z ,.?!'-]", "", text): return None
    tokens = tuple(WORD.findall(text.lower())); expected = slots(grammar.expand(grammar.start()))
    if len(tokens) != len(expected): return None
    for i, slot in enumerate(expected):
        if not any(tokens[i] == item.word for item in choices(slot.symbol)): return None
        if slot.symbol.kind == "det" and not determiner_ok(tokens[i], tokens[i+1]): return None
    verbs = {x.word for x in BY_CATEGORY["verb"] if x.subject == "person" and x.object == "artifact"}
    if tokens[3] not in verbs: return None
    return grammar.expand(grammar.start())


def audit(grammar: CompoundFeatureGrammar, text: str, kind: str, provenance: tuple[str, ...]) -> dict[str, object]:
    exact = exact_audit(text); parsed = parse_tree(grammar, text); central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [key for key, value in central.items() if not value]
    if parsed is None: codes.append("independent_complete_reparse_failed")
    return {"record_kind": kind, "rendered": text, "provenance": provenance, "independent_exact_audit": exact,
            "independent_parse": parsed is not None, "central_admission": central, "mechanically_admitted": not codes,
            "rejection_codes": codes, "reader_status": "unreviewed; programmatic checks do not certify readability"}


def center_out_search(leaf_slots: tuple[Slot, ...], grammar: CompoundFeatureGrammar, *, state_limit: int, closure_limit: int, stats: Counter) -> list[dict[str, object]]:
    if len(leaf_slots) % 2: raise ValueError("center bridge grammar must have an even leaf frontier")
    rows: list[dict[str, object]] = []; words = [""] * len(leaf_slots); center = len(leaf_slots)//2

    def visit(lo: int, hi: int, assigned: dict[str, Lexeme], left_residual: str, right_residual: str, owner: int, last_side: int) -> None:
        if len(rows) >= closure_limit or stats["state_count"] >= state_limit:
            if stats["state_count"] >= state_limit: stats["state_limit_reached"] += 1
            return
        if lo < 0 and hi >= len(leaf_slots):
            stats["complete_tree_states"] += 1; residual = left_residual or right_residual
            if residual and residual != residual[::-1]: stats["central_residual_rejections"] += 1; return
            row = audit(grammar, render(words), "complete_center_bridge_closure", tuple(words))
            if row["mechanically_admitted"]: rows.append(row)
            else: stats["mechanical_rejections"] += 1
            return
        if left_residual: side = -1 if hi < len(leaf_slots) else 0
        elif right_residual: side = 1 if lo >= 0 else 0
        else:
            side = 1 if last_side in {0, -1} and lo >= 0 else -1
            if side == -1 and hi >= len(leaf_slots): side = 1
        if side == 0: stats["unmatched_frontier_rejections"] += 1; return
        index = lo if side == 1 else hi; slot = leaf_slots[index]
        for item in choices(slot.symbol):
            if not semantic_ok(slot, item, assigned): continue
            stats["state_count"] += 1
            updated = emit_left_outward(left_residual, right_residual, item.word, stats) if side == 1 else emit_right_outward(left_residual, right_residual, item.word, stats)
            if updated is None: continue
            assigned[slot.symbol.role] = item; words[index] = item.word
            if side == 1: stats["left_leaf_emissions"] += 1
            else: stats["right_leaf_emissions"] += 1
            visit(lo - (side == 1), hi + (side == -1), assigned, updated[0], updated[1], 1 if updated[0] else (-1 if updated[1] else 0), side)
            words[index] = ""; assigned.pop(slot.symbol.role, None)
    visit(center-1, center, {}, "", "", 0, 0); return rows


def run(*, state_limit: int = 100_000, closure_limit: int = 100) -> dict[str, object]:
    grammar = CompoundFeatureGrammar(); tree = grammar.expand(grammar.start()); leaf_slots = slots(tree)
    stats = Counter(state_count=0, left_char_emissions=0, right_char_emissions=0, left_leaf_emissions=0, right_leaf_emissions=0,
                    residual_cancellations=0, residual_contradictions=0, complete_tree_states=0, central_residual_rejections=0, mechanical_rejections=0)
    closures = center_out_search(leaf_slots, grammar, state_limit=state_limit, closure_limit=closure_limit, stats=stats)
    controls = (
        "An expert researcher reviews a comprehensive substantial manuscript record after the inspection near a repository today.",
        "A patient archivist catalogs a comprehensive substantial collection artifact after the inspection near a repository today.",
    )
    control_rows = [audit(grammar, text, "complete_connected_grammar_control", tuple(WORD.findall(text.lower()))) for text in controls]
    return {"status": "center_bridge_compound_tree_single_tree", "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS,
        "state_limit": state_limit, "closure_limit": closure_limit, "one_connected_tree": True, "grammar_owns_every_leaf": True,
        "center_out_exposed_slot_schedule": True, "character_residual_per_emission": True, "multi_character_center_bridge": True,
        "closure_requires_complete_tree": True, "independent_complete_reparse": True, "corpus_or_catalogue_generation": False},
        "center_bridge_feasibility": center_bridge_examples(leaf_slots), "stats": dict(stats), "exact_closures": closures,
        "complete_grammar_controls": control_rows, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(), "material": "task-authored compound feature grammar and lexicon; no corpus/catalogue text"},
        "reader_facing_next_operator": "Retain the multi-character compound bridge but add a semantic compatibility table for artifact compounds before expanding vocabulary; continue to require complete-tree independent reparse.",
        "scope": "Exactness and feature parsing are mechanical filters; readability requires blinded human evidence."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--state-limit", type=int, default=100_000); parser.add_argument("--closure-limit", type=int, default=100)
    args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit, closure_limit=args.closure_limit); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["state_count"], "closures": len(result["exact_closures"]), "bridge_count": result["center_bridge_feasibility"]["bridge_count"]}, indent=2))


if __name__ == "__main__": main()
