"""Cross-leaf residual construction with a non-palindromic center node.

The center grammar node owns two independently selected leaves (a verb and an
adjective).  It is deliberately *not* a palindrome.  If its live residual
survives the node boundary, the scheduler must expose neighboring leaves and
cancel it character by character.  A hard generator constraint rejects every
self-palindromic contiguous multiword span before any closure is counted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import itertools
import json
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

WORD = re.compile(r"[a-z]+")
MIN_LETTERS, MAX_LETTERS = 100, 220


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


def T(kind: str, role: str) -> Symbol: return Symbol("T", role, kind)


LEXICON = (
    Lexeme("a", "det", "indefinite"), Lexeme("an", "det", "indefinite"), Lexeme("the", "det", "definite"),
    Lexeme("during", "prep", "temporal"), Lexeme("after", "prep", "temporal"), Lexeme("for", "prep", "benefactive"),
    Lexeme("near", "prep", "locative"), Lexeme("with", "prep", "accompaniment"), Lexeme("in", "prep", "locative"),
    Lexeme("long", "adj_event", "duration"), Lexeme("detailed", "adj_event", "event"), Lexeme("careful", "adj_person", "person"),
    Lexeme("young", "adj_person", "person"), Lexeme("patient", "adj_person", "person"), Lexeme("diligent", "adj_person", "person"),
    Lexeme("red", "adj_artifact", "color"), Lexeme("hot", "adj_artifact", "temperature"),
    Lexeme("trial", "noun_event", "event"), Lexeme("study", "noun_event", "event"), Lexeme("chef", "noun_person", "person"),
    Lexeme("editor", "noun_person", "person"), Lexeme("researcher", "noun_person", "person"), Lexeme("archive", "noun_org", "organization"),
    Lexeme("museum", "noun_org", "organization"), Lexeme("root", "noun_artifact", "food"), Lexeme("meal", "noun_artifact", "food"),
    Lexeme("herb", "noun_artifact", "food"), Lexeme("to", "inf", "infinitive"), Lexeme("also", "adv", "addition"),
    Lexeme("deliberately", "adv", "manner"), Lexeme("today", "adv", "temporal"),
    Lexeme("decided", "verb", "decision", "person", "event"), Lexeme("eat", "verb", "eating", "person", "food"),
    Lexeme("order", "verb", "ordering", "person", "food"),
)
BY_KIND: dict[str, tuple[Lexeme, ...]] = {}
for item in LEXICON:
    BY_KIND.setdefault(item.kind, ()); BY_KIND[item.kind] += (item,)


class Grammar:
    """A single S tree with a two-leaf, non-palindromic CENTER node."""
    def productions(self, symbol: Symbol) -> tuple[tuple[Symbol, ...], ...]:
        if symbol.name == "S":
            return ((Symbol("NP", "subject", "person"), Symbol("PP", "during", "event_pp"), Symbol("PP", "after", "event_pp"),
                     T("adv", "manner"), T("adv", "addition"), T("verb", "matrix_verb"), T("inf", "bridge_left"),
                     Symbol("CENTER", "center", "nonpal"), T("noun_artifact", "bridge_right"), Symbol("PP", "for", "benefactive"),
                     Symbol("PP", "near", "locative"), Symbol("PP", "with", "accompaniment"), Symbol("PP", "in", "locative"),
                     T("adv", "today")),)
        if symbol.name == "CENTER":
            return ((T("verb", "center_verb"), T("adj_artifact", "center_adj")),)
        if symbol.name == "NP":
            return ((T("det", symbol.role + "_det"), T("adj_person", symbol.role + "_adj1"), T("adj_person", symbol.role + "_adj2"), T("noun_person", symbol.role + "_noun")),)
        if symbol.name == "PP":
            noun = "noun_event" if symbol.kind == "event_pp" else ("noun_org" if symbol.role in {"near", "in"} else "noun_person")
            adj = "adj_event" if symbol.role in {"during", "after"} else "adj_person"
            if symbol.role in {"near", "in"}:
                return ((T("prep", symbol.role), T("det", symbol.role + "_det"), T(noun, symbol.role + "_noun")),)
            return ((T("prep", symbol.role), T("det", symbol.role + "_det"), T(adj, symbol.role + "_adj"), T(noun, symbol.role + "_noun")),)
        return ()

    def expand(self, symbol: Symbol) -> Node:
        rhs = self.productions(symbol); return Node(symbol, tuple(self.expand(child) for child in rhs[0])) if rhs else Node(symbol)

    def digest(self) -> str:
        rows, seen, queue = [], set(), [Symbol("S")]
        while queue:
            symbol = queue.pop(0)
            if symbol in seen: continue
            seen.add(symbol); rhs = self.productions(symbol)
            rows.append((symbol.name, symbol.role, symbol.kind, [[(x.name, x.role, x.kind) for x in row] for row in rhs]))
            queue.extend(x for row in rhs for x in row if x.name != "T")
        return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def slots(tree: Node) -> tuple[Slot, ...]:
    result: list[Slot] = []
    def walk(node: Node) -> None:
        if node.symbol.name == "T": result.append(Slot(len(result), node.symbol))
        else:
            for child in node.children: walk(child)
    walk(tree); return tuple(result)


def normalized_span(words: Iterable[str]) -> str: return normalize_letters(" ".join(words))


def self_palindromic_multiword_spans(words: tuple[str, ...]) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    for start in range(len(words)):
        for end in range(start + 2, len(words) + 1):
            tape = normalized_span(words[start:end])
            if tape and tape == tape[::-1]: spans.append({"start": start, "end": end, "words": words[start:end], "normalized": tape})
    return spans


def crossleaf_trace(left_center: str, right_center: str, left_outer: str, right_outer: str) -> dict[str, object]:
    """Cross CENTER's residual into adjacent leaves without whole-word matching."""
    leaves = {"left": [(left_center, left_center[::-1]), (left_outer, left_outer[::-1])],
              "right": [(right_center, right_center), (right_outer, right_outer)]}
    cursor = {"left": [0, 0], "right": [0, 0]}; current = {"left": None, "right": None}
    residual = ""; owner = ""; preferred = "left"; trace: list[dict[str, object]] = []; cancellations = 0
    while True:
        for side in ("left", "right"):
            if current[side] is not None and cursor[side][1] >= len(current[side][1]): current[side] = None
        if all(current[s] is None and cursor[s][0] >= len(leaves[s]) for s in ("left", "right")): break
        if residual: side = "right" if owner == "left" else "left"
        elif preferred == "left" and (current["left"] is not None or cursor["left"][0] < len(leaves["left"])): side = "left"
        else: side = "right"
        if current[side] is None:
            if cursor[side][0] >= len(leaves[side]): return {"completed": False, "trace": trace, "cancellations": cancellations}
            index = cursor[side][0]; current[side] = leaves[side][index]; cursor[side] = [index + 1, 0]
        word, stream = current[side]; char = stream[cursor[side][1]]; cursor[side][1] += 1; before = residual
        if residual and char != residual[0]:
            trace.append({"side": side, "word": word, "char": char, "residual_before": before, "action": "contradiction"})
            return {"completed": False, "trace": trace, "cancellations": cancellations}
        if residual: residual = residual[1:]; cancellations += 1; action = "cancel"
        else: residual = char; owner = side; action = "open"
        trace.append({"side": side, "word": word, "char": char, "residual_before": before, "residual_after": residual, "action": action})
        if not residual: preferred = "left"
    return {"completed": not residual, "trace": trace, "cancellations": cancellations,
            "leaves": {"left": (left_center, left_outer), "right": (right_center, right_outer)}}


def semantically_licensed(center: tuple[Lexeme, Lexeme], outer_left: Lexeme, outer_right: Lexeme) -> bool:
    verb, adjective = center
    return (verb.subject_type == "person" and verb.object_type == outer_right.meaning and adjective.meaning == "color" and
            outer_left.meaning == "infinitive" and outer_left.word == "to" and outer_right.meaning == "food")


def discover_cross_bridge(stats: Counter) -> dict[str, object] | None:
    channels = (BY_KIND["verb"], BY_KIND["adj_artifact"], BY_KIND["inf"], BY_KIND["noun_artifact"])
    for verb, adjective, outer_left, outer_right in itertools.product(*channels):
        stats["independent_assignments"] += 1
        if not semantically_licensed((verb, adjective), outer_left, outer_right): continue
        stats["semantically_licensed_assignments"] += 1
        trace = crossleaf_trace(verb.word, adjective.word, outer_left.word, outer_right.word); stats["crossleaf_traces"] += 1
        if trace["completed"]:
            stats["crossleaf_complete"] += 1
            words = (outer_left.word, verb.word, adjective.word, outer_right.word)
            spans = self_palindromic_multiword_spans(words)
            return {"words": words, "center_words": (verb.word, adjective.word), "center_normalized": normalized_span((verb.word, adjective.word)),
                    "center_is_self_palindrome": normalized_span((verb.word, adjective.word)) == normalized_span((verb.word, adjective.word))[::-1],
                    "trace": trace, "self_palindromic_spans": spans, "hard_generator_reject": bool(spans)}
    return None


def exact_audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text); mismatches = [(i, len(tape) - 1 - i) for i in range(len(tape) // 2) if tape[i] != tape[-i - 1]]
    return {"exact": bool(tape) and not mismatches, "letters": len(tape), "mismatches": mismatches, "normalized_sha256": sha256(tape.encode()).hexdigest()}


def parse_tree(grammar: Grammar, text: str) -> bool:
    if text != text.strip() or re.sub(r"[A-Za-z ,.?!'-]", "", text): return False
    tokens = tuple(WORD.findall(text.lower())); leaf_slots = slots(grammar.expand(Symbol("S")))
    if len(tokens) != len(leaf_slots): return False
    assigned: list[Lexeme] = []
    for i, slot in enumerate(leaf_slots):
        item = next((x for x in BY_KIND.get(slot.symbol.kind, ()) if x.word == tokens[i]), None)
        if item is None: return False
        assigned.append(item)
        if item.kind == "det" and item.word in {"a", "an"} and ((item.word == "an") != (tokens[i + 1][0] in "aeiou")): return False
    return (assigned[3].meaning == "person" and assigned[14].subject_type == "person" and
            assigned[16].subject_type == "person" and assigned[17].meaning == "color" and assigned[18].meaning == "food")


def audit(grammar: Grammar, text: str, kind: str, provenance: tuple[str, ...]) -> dict[str, object]:
    exact = exact_audit(text); parsed = parse_tree(grammar, text); gate = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [key for key, value in gate.items() if not value]
    spans = self_palindromic_multiword_spans(tuple(WORD.findall(text.lower())))
    if spans: codes.append("self_palindromic_multiword_span")
    if not parsed: codes.append("independent_complete_reparse_failed")
    return {"record_kind": kind, "rendered": text, "provenance": provenance, "independent_exact_audit": exact, "independent_parse": parsed,
            "self_palindromic_multiword_spans": spans, "central_admission": gate, "mechanically_admitted": not codes,
            "rejection_codes": codes, "reader_status": "unreviewed; programmatic checks do not certify readability"}


def render(words: Iterable[str]) -> str:
    text = " ".join(words); return text[:1].upper() + text[1:] + "."


def run(*, state_limit: int = 100_000, closure_limit: int = 100) -> dict[str, object]:
    grammar = Grammar(); leaf_slots = slots(grammar.expand(Symbol("S"))); stats = Counter(state_count=0)
    found = discover_cross_bridge(stats)
    if found is None: raise RuntimeError("typed channels found no cross-leaf trace")
    stats["state_count"] += len(found["trace"]["trace"])
    prefix = "the careful young chef during the long trial after the detailed study deliberately also decided".split()
    suffix = "for the patient editor near the archive with a diligent researcher in the museum today".split()
    words = prefix + list(found["words"]) + suffix
    controls = [audit(grammar, render(words), "complete_connected_grammar_control", tuple(words)),
                audit(grammar, render("the patient diligent chef during the detailed trial after the long study deliberately also decided".split() + list(found["words"]) + suffix), "complete_connected_grammar_control", tuple(words))]
    return {"status": "nonpal_center_crossleaf_residual_single_tree", "config": {"state_limit": state_limit, "closure_limit": closure_limit,
        "one_connected_tree": True, "grammar_owns_every_leaf": True, "nonpalindromic_center_node": True,
        "live_residual_crosses_center_boundary": True, "one_character_emission_states": True,
        "reject_every_self_palindromic_contiguous_multiword_span": True, "corpus_or_catalogue_generation": False},
        "grammar_leaf_count": len(leaf_slots), "discovered_cross_bridge": found, "stats": dict(stats),
        "exact_closures": [], "admitted_closures": [], "complete_grammar_controls": controls,
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(),
                       "material": "task-authored connected grammar and independent typed lexical channels; no prebuilt palindrome phrase or corpus/catalogue text"},
        "reader_facing_next_operator": "Repair the outer typed frontier while retaining the hard contiguous-span rejection; only a full exact tree with no rejected span may proceed to reader testing.",
        "scope": "The cross-leaf trace is a rejected construction diagnostic; no candidate or readability claim is made."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--state-limit", type=int, default=100_000); parser.add_argument("--closure-limit", type=int, default=100); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit, closure_limit=args.closure_limit); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["state_count"], "exact": len(result["exact_closures"]), "admitted": len(result["admitted_closures"]), "hard_reject": result["discovered_cross_bridge"]["hard_generator_reject"]}, indent=2))


if __name__ == "__main__": main()
