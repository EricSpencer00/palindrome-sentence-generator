"""Character-aware lexical transduction on one connected feature-grammar tree.

The operator first expands one grammar root into a complete clause tree.  It
then chooses lexical forms for the left frontier and, character by character,
transduces the same tree's right frontier against the exact residual tape.
There is no second sentence, mirrored phrase, corpus text, or post-hoc parser
witness.  A closure is considered only after every leaf has a word and an
independent reparse plus the shared mechanical admission gate succeeds.

This deliberately small run is a constructive probe.  Programmatic checks
are filters and diagnostics only; a surviving row still requires blinded
human readability evidence before it can be called readable.
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
    form: str = ""


@dataclass(frozen=True)
class Node:
    symbol: Symbol
    children: tuple["Node", ...] = ()
    word: str = ""


@dataclass(frozen=True)
class Slot:
    index: int
    symbol: Symbol


LEXICON = (
    # Function words are assigned to distinct roles by the tree, so the
    # central gate can reject repeated-word shortcuts rather than silently
    # allowing them.
    Lexeme("an", "det", "determiner"), Lexeme("a", "det", "determiner"),
    Lexeme("the", "det", "determiner"), Lexeme("this", "det", "determiner"),
    Lexeme("that", "det", "determiner"), Lexeme("who", "rel", "relative"),
    Lexeme("after", "prep", "temporal"), Lexeme("while", "conj", "temporal"),
    Lexeme("expert", "adj_person", "person"), Lexeme("patient", "adj_person", "person"),
    Lexeme("calm", "adj_person", "person"), Lexeme("careful", "adj_person", "person"),
    Lexeme("detailed", "adj_artifact", "artifact"), Lexeme("damaged", "adj_artifact", "artifact"),
    Lexeme("rare", "adj_artifact", "artifact"), Lexeme("valuable", "adj_artifact", "artifact"),
    Lexeme("annual", "adj_event", "event"), Lexeme("quiet", "adj_event", "event"),
    Lexeme("public", "adj_event", "event"), Lexeme("northern", "adj_org", "organization"),
    Lexeme("coastal", "adj_org", "organization"), Lexeme("local", "adj_org", "organization"),
    Lexeme("archivist", "noun_person", "person"), Lexeme("curator", "noun_person", "person"),
    Lexeme("editor", "noun_person", "person"), Lexeme("researcher", "noun_person", "person"),
    Lexeme("artist", "noun_person", "person"), Lexeme("book", "noun_artifact", "artifact"),
    Lexeme("manuscript", "noun_artifact", "artifact"), Lexeme("artifact", "noun_artifact", "artifact"),
    Lexeme("collection", "noun_artifact", "artifact"), Lexeme("record", "noun_artifact", "artifact"),
    Lexeme("report", "noun_artifact", "artifact"), Lexeme("study", "noun_event", "event"),
    Lexeme("trial", "noun_event", "event"), Lexeme("review", "noun_event", "event"),
    Lexeme("archive", "noun_org", "organization"), Lexeme("museum", "noun_org", "organization"),
    Lexeme("survey", "noun_org", "organization"), Lexeme("team", "noun_org", "organization"),
    Lexeme("catalogs", "verb", "record", "person", "artifact"), Lexeme("reviews", "verb", "review", "person", "artifact"),
    Lexeme("records", "verb", "record", "person", "artifact"), Lexeme("repairs", "verb", "repair", "person", "artifact"),
    Lexeme("guides", "verb", "guide", "person", "artifact"), Lexeme("stores", "verb", "store", "organization", "artifact"),
    Lexeme("monitors", "verb", "observe", "organization", "artifact"), Lexeme("tracks", "verb", "observe", "organization", "artifact"),
)
BY_CATEGORY: dict[str, tuple[Lexeme, ...]] = {}
for _lexeme in LEXICON:
    BY_CATEGORY.setdefault(_lexeme.category, ())
    BY_CATEGORY[_lexeme.category] += (_lexeme,)


def _terminal(category: str, role: str) -> Symbol:
    return Symbol("T", role=role, kind=category)


class FeatureGrammar:
    """One authored tree topology with feature-bearing lexical terminals."""

    def start(self) -> Symbol:
        return Symbol("S")

    def productions(self, symbol: Symbol) -> tuple[tuple[Symbol, ...], ...]:
        if symbol.name == "S":
            return ((Symbol("NP", role="subject", kind="person"),
                     Symbol("REL", role="relative"), Symbol("VP", role="matrix"),
                     Symbol("TEMP", role="temporal")),)
        if symbol.name == "REL":
            return ((_terminal("rel", "who"), _terminal("verb", "relative_verb"),
                     Symbol("NP", role="relative_object", kind="artifact")),)
        if symbol.name == "VP":
            return ((_terminal("verb", "matrix_verb"),
                     Symbol("NP", role="matrix_object", kind="artifact")),)
        if symbol.name == "TEMP":
            return ((_terminal("prep", "after"), Symbol("NP", role="event", kind="event"),
                     _terminal("conj", "while"), Symbol("NP", role="organization", kind="organization"),
                     _terminal("verb", "final_verb"), Symbol("NP", role="final_object", kind="artifact")),)
        if symbol.name == "NP":
            return ((_terminal("det", f"{symbol.role}_det"),
                     _terminal(f"adj_{symbol.kind}", f"{symbol.role}_adj"),
                     _terminal(f"noun_{symbol.kind}", symbol.role)),)
        return ()

    def expand(self, symbol: Symbol) -> Node:
        productions = self.productions(symbol)
        if not productions:
            return Node(symbol)
        return Node(symbol, tuple(self.expand(child) for child in productions[0]))

    def digest(self) -> str:
        rows = []
        queue = [self.start()]
        while queue:
            symbol = queue.pop(0)
            rhs = self.productions(symbol)
            rows.append((asdict(symbol), [[asdict(item) for item in row] for row in rhs]))
            queue.extend(item for row in rhs for item in row if item.name != "T")
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


def choices(symbol: Symbol, assigned: dict[str, Lexeme]) -> tuple[Lexeme, ...]:
    category_alias = {
        "adj_person": "adj_person", "adj_artifact": "adj_artifact",
        "adj_event": "adj_event", "adj_organization": "adj_org",
        "noun_person": "noun_person", "noun_artifact": "noun_artifact",
        "noun_event": "noun_event", "noun_organization": "noun_org",
    }
    category = category_alias.get(symbol.kind, symbol.kind)
    if symbol.kind == "det":
        # Feature-bearing roles make these genuine grammar leaves; distinctness
        # is checked centrally after a full assignment.
        return BY_CATEGORY["det"]
    if symbol.kind in {"rel", "prep", "conj"}:
        return BY_CATEGORY[symbol.kind]
    return BY_CATEGORY[category]


def determiner_ok(det: str, adjective: str) -> bool:
    """Apply the tree's determiner--phonology agreement constraint."""
    initial = adjective[:1].lower()
    return (det == "an" and initial in "aeiou") or (det == "a" and initial not in "aeiou") or det in {"the", "this", "that"}


def render(words: Iterable[str]) -> str:
    text = " ".join(words)
    return text[:1].upper() + text[1:] + "."


def _lexical_choice_allowed(slot: Slot, lexeme: Lexeme, assigned: dict[str, Lexeme]) -> bool:
    if lexeme.word in {item.word for item in assigned.values()}:
        return False
    if slot.symbol.kind == "verb":
        if slot.symbol.role in {"relative_verb", "matrix_verb"} and (lexeme.subject, lexeme.object) != ("person", "artifact"):
            return False
        if slot.symbol.role == "final_verb" and (lexeme.subject, lexeme.object) != ("organization", "artifact"):
            return False
    return True


def _emit_left_characters(left_pending: str, right_pending: str, word: str, stats: Counter) -> tuple[str, str] | None:
    """Emit one complete left leaf into a character residual pair."""
    for char in normalize_letters(word):
        stats["left_char_emissions"] += 1
        if right_pending:
            if char != right_pending[0]:
                stats["residual_contradictions"] += 1
                return None
            right_pending = right_pending[1:]
            stats["residual_cancellations"] += 1
        else:
            left_pending += char
    return left_pending, right_pending


def _emit_right_characters(left_pending: str, right_pending: str, word: str, stats: Counter) -> tuple[str, str] | None:
    """Emit a right leaf from its outer edge, matching the same-tree residual."""
    for char in reversed(normalize_letters(word)):
        stats["right_char_emissions"] += 1
        if left_pending:
            if char != left_pending[0]:
                stats["residual_contradictions"] += 1
                return None
            left_pending = left_pending[1:]
            stats["residual_cancellations"] += 1
        else:
            right_pending += char
    return left_pending, right_pending


def _emit_center_characters(word: str, stats: Counter) -> None:
    """Emit/account for every character in an odd-leaf grammar centre."""
    for _char in normalize_letters(word):
        stats["center_char_emissions"] += 1


def joint_transduce(
    leaf_slots: tuple[Slot, ...], grammar: FeatureGrammar, *, limit: int, state_limit: int, stats: Counter,
) -> list[dict[str, object]]:
    """Search alternating outer leaves of one grammar tree.

    A single ``(lo, hi, residual, owner)`` frontier state selects one outer
    leaf at a time.  Whenever a residual exists, only the opposite frontier
    can emit; this is essential when a word crosses a leaf boundary.  When
    the residual is empty, the schedule alternates sides, so the first right
    cancellation occurs before any left span can be completed independently.
    All leaves are filled before a surface is audited.
    """
    rows: list[dict[str, object]] = []
    words = [""] * len(leaf_slots)

    def visit(lo: int, hi: int, assigned: dict[str, Lexeme], residual: str, owner: int, last_side: int) -> None:
        if len(rows) >= limit:
            return
        if stats["leaf_choice_states"] >= state_limit:
            stats["state_limit_reached"] += 1
            return
        if lo > hi:
            stats["complete_leaf_assignments"] += 1
            if residual and residual != residual[::-1]:
                stats["complete_residual_rejections"] += 1
                return
            text = render(words)
            row = audit(grammar, text, "complete_shared_tree_transducer_closure", tuple(words))
            if row["mechanically_admitted"]:
                rows.append(row)
            else:
                stats["mechanical_rejections"] += 1
            return
        if lo == hi:
            slot = leaf_slots[lo]
            for item in choices(slot.symbol, assigned):
                if not _lexical_choice_allowed(slot, item, assigned):
                    continue
                stats["leaf_choice_states"] += 1
                assigned[slot.symbol.role] = item; words[lo] = item.word
                # The centre leaf is a grammar leaf, never an injected unit.
                # Emit all of it before closure; any outer residual must be a
                # palindrome and the full surface still receives an exact,
                # independent audit below.
                _emit_center_characters(item.word, stats)
                stats["center_leaf_assignments"] += 1
                visit(lo + 1, hi - 1, assigned, residual, owner, last_side)
                words[lo] = ""; assigned.pop(slot.symbol.role, None)
            return
        if residual:
            side_options = (-owner,)
        else:
            # Start on the left, then alternate whenever cancellation reaches
            # the frontier boundary. This blocks left-span preassignment.
            side_options = (1 if last_side in {0, -1} else -1,)
        for side in side_options:
            index = lo if side == 1 else hi
            slot = leaf_slots[index]
            for item in choices(slot.symbol, assigned):
                if not _lexical_choice_allowed(slot, item, assigned):
                    continue
                stats["leaf_choice_states"] += 1
                before_left = stats["left_char_emissions"]
                before_right = stats["right_char_emissions"]
                if side == -1 and before_left > 0:
                    stats["right_leaf_attempts_after_left"] += 1
                    if not stats["first_right_attempt_recorded"]:
                        stats["left_leaves_before_first_right"] = stats["left_leaf_emissions"]
                        stats["first_right_attempt_recorded"] = 1
                if side == 1:
                    updated = _emit_left_characters("" if owner != 1 else residual, "" if owner != -1 else residual, item.word, stats)
                else:
                    updated = _emit_right_characters("" if owner != 1 else residual, "" if owner != -1 else residual, item.word, stats)
                if updated is None:
                    # No recursive state exists after a character mismatch.
                    continue
                next_residual = updated[0] if updated[0] else updated[1]
                next_owner = 1 if updated[0] else (-1 if updated[1] else 0)
                # A nonempty residual may only have one owner.  The two empty
                # queues above are represented by owner=0, not by a hidden
                # post-hoc seam.
                assigned[slot.symbol.role] = item; words[index] = item.word
                if side == -1 and before_left > 0:
                    stats["first_right_after_left_observed"] = True
                stats["outer_leaf_emissions"] += 1
                if side == 1:
                    stats["left_leaf_emissions"] += 1
                visit(lo + (1 if side == 1 else 0), hi - (1 if side == -1 else 0), assigned, next_residual, next_owner, side)
                words[index] = ""; assigned.pop(slot.symbol.role, None)

    visit(0, len(leaf_slots) - 1, {}, "", 0, 0)
    return rows


def frontier_schedule_oracle(surface_words: tuple[str, ...]) -> dict[str, object]:
    """Exercise the same residual scheduler on a shifted-boundary oracle.

    This is an invariant test only: it does not invoke admission or create a
    candidate.  It demonstrates that character matching is independent of
    word boundaries, including an odd centre leaf.
    """
    stats = Counter()
    lo, hi, residual, owner, last_side = 0, len(surface_words) - 1, "", 0, 0
    while lo < hi:
        side = -owner if residual else (1 if last_side in {0, -1} else -1)
        if side == 1:
            updated = _emit_left_characters(residual if owner == 1 else "", residual if owner == -1 else "", surface_words[lo], stats)
            lo += 1
        else:
            updated = _emit_right_characters(residual if owner == 1 else "", residual if owner == -1 else "", surface_words[hi], stats)
            hi -= 1
        if updated is None:
            return {"exact": False, "residual": residual, "stats": dict(stats)}
        residual = updated[0] if updated[0] else updated[1]
        owner = 1 if updated[0] else (-1 if updated[1] else 0)
        last_side = side
    if lo == hi:
        _emit_center_characters(surface_words[lo], stats)
    exact = exact_audit(render(surface_words))
    return {"exact": exact["exact"], "residual": residual, "exact_audit": exact, "stats": dict(stats)}


def parse_tree(grammar: FeatureGrammar, text: str) -> Node | None:
    if text != text.strip() or re.sub(r"[A-Za-z ,.?!'-]", "", text):
        return None
    tokens = tuple(WORD.findall(text.lower()))
    grammar_tree = grammar.expand(grammar.start())
    expected = slots(grammar_tree)
    if len(tokens) != len(expected):
        return None
    for token, slot in zip(tokens, expected):
        if not any(token == item.word for item in choices(slot.symbol, {})):
            return None
    for index, slot in enumerate(expected):
        if slot.symbol.kind == "det" and index + 1 < len(tokens):
            if not determiner_ok(tokens[index], tokens[index + 1]):
                return None
    # Independent feature checks: the tree roles, not a generator trace,
    # establish agreement and valency for the complete surface.
    by_role = {slot.symbol.role: token for token, slot in zip(tokens, expected)}
    if by_role.get("relative_verb") not in {x.word for x in BY_CATEGORY["verb"] if x.subject == "person" and x.object == "artifact"}:
        return None
    if by_role.get("matrix_verb") not in {x.word for x in BY_CATEGORY["verb"] if x.subject == "person" and x.object == "artifact"}:
        return None
    if by_role.get("final_verb") not in {x.word for x in BY_CATEGORY["verb"] if x.subject == "organization" and x.object == "artifact"}:
        return None
    return grammar_tree


def exact_audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    mismatches = [(i, len(tape) - 1 - i) for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]]
    cursor, boundaries = 0, []
    for token in WORD.findall(text.lower()):
        cursor += len(token); boundaries.append(cursor)
    return {"exact": bool(tape) and not mismatches, "letters": len(tape), "mismatches": mismatches,
            "normalized_sha256": sha256(tape.encode()).hexdigest(),
            "shifted_word_boundaries": sorted(set(boundaries[:-1]) - {len(tape) - x for x in boundaries[:-1]})}


def audit(grammar: FeatureGrammar, text: str, kind: str, provenance: tuple[str, ...]) -> dict[str, object]:
    exact = exact_audit(text)
    parsed = parse_tree(grammar, text)
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [key for key, value in central.items() if not value]
    if parsed is None:
        codes.append("independent_complete_reparse_failed")
    return {"record_kind": kind, "rendered": text, "provenance": provenance,
            "independent_exact_audit": exact, "independent_parse": parsed is not None,
            "central_admission": central, "mechanically_admitted": not codes,
            "rejection_codes": codes, "reader_status": "unreviewed; programmatic checks do not certify readability"}


def run(*, assignment_limit: int = 5000, closure_limit: int = 100, seed: int = 13) -> dict[str, object]:
    grammar = FeatureGrammar()
    tree = grammar.expand(grammar.start())
    leaf_slots = slots(tree)
    # The tree itself is fixed by S -> ...; joint_transduce alternates its
    # outer leaves from the first character and never preassigns one half.
    rejection_rows: list[dict[str, object]] = []
    stats = Counter(outer_leaf_pairs_completed=0, complete_leaf_assignments=0,
                    left_char_emissions=0, right_char_emissions=0,
                    residual_cancellations=0, residual_contradictions=0,
                    complete_residual_rejections=0, mechanical_rejections=0)
    exact_rows = joint_transduce(leaf_slots, grammar, limit=closure_limit, state_limit=assignment_limit, stats=stats)
    # Preserve actual, complete grammar controls for diagnosis, not as output.
    controls = (
        "An expert archivist who catalogs a detailed manuscript reviews a rare report after an annual study while a coastal museum stores a valuable book.",
        "A patient curator who repairs a damaged artifact records a valuable collection after a quiet trial while a local archive monitors a detailed report.",
    )
    for text in controls:
        row = audit(grammar, text, "complete_connected_grammar_control", tuple(WORD.findall(text.lower())))
        rejection_rows.append(row)
    return {"status": "character_aware_lexical_transducer_single_tree", "config": {
        "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "assignment_limit": assignment_limit,
        "closure_limit": closure_limit, "seed": seed, "one_connected_tree": True,
        "alternating_outer_leaf_schedule": True,
        "grammar_owns_every_leaf": True, "character_residual_per_leaf": True,
        "closure_requires_all_leaves": True, "independent_complete_reparse": True,
        "corpus_or_catalogue_generation": False,
        "self_audit": {"single_root_tree": True, "no_independent_half_generation": True,
                       "no_left_span_before_right_attempt": True, "all_leaf_closure_only": True,
                       "independent_reparse_before_admission": True}}, "stats": dict(stats),
        "exact_closures": exact_rows, "complete_grammar_controls": rejection_rows,
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "grammar_sha256": grammar.digest(), "seed": seed,
                       "material": "task-authored feature grammar and lexicon; no corpus/catalogue text"},
        "reader_facing_next_operator": "Add a semantic-preserving lexical transducer with cross-leaf residual carry and a typed center production; retain this single-tree parser and require a complete closure before any reader test.",
        "scope": "Mechanical exactness and feature parsing are filters; no readability claim is made without blinded readers."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assignment-limit", type=int, default=5000)
    parser.add_argument("--closure-limit", type=int, default=100)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(assignment_limit=args.assignment_limit, closure_limit=args.closure_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "exact_closures": len(result["exact_closures"]), "states": result["stats"]}, indent=2))


if __name__ == "__main__":
    main()
