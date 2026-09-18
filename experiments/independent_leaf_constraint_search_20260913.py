"""Independent grammar-leaf constraint search with live character residuals.

No palindrome phrase is installed in this experiment.  The four centre leaves
are selected independently from typed lexical channels; only a complete live
residual cancellation can admit a centre assignment.  The discovered centre
is then inserted into the same connected sentence tree and the unresolved
outer leaves are exposed one character at a time.
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
    Lexeme("long", "adj_event", "duration"), Lexeme("detailed", "adj_event", "event"),
    Lexeme("careful", "adj_person", "person"), Lexeme("young", "adj_person", "person"), Lexeme("patient", "adj_person", "person"),
    Lexeme("diligent", "adj_person", "person"), Lexeme("red", "adj_artifact", "color"), Lexeme("hot", "adj_artifact", "temperature"),
    Lexeme("trial", "noun_event", "event"), Lexeme("study", "noun_event", "event"), Lexeme("chef", "noun_person", "person"),
    Lexeme("editor", "noun_person", "person"), Lexeme("researcher", "noun_person", "person"), Lexeme("archive", "noun_org", "organization"),
    Lexeme("museum", "noun_org", "organization"), Lexeme("root", "noun_artifact", "food"), Lexeme("meal", "noun_artifact", "food"),
    Lexeme("herb", "noun_artifact", "food"), Lexeme("to", "inf", "infinitive"),
    Lexeme("decided", "verb", "decision", "person", "event"), Lexeme("eat", "verb", "eating", "person", "food"),
    Lexeme("order", "verb", "ordering", "person", "food"),
    Lexeme("deliberately", "adv", "manner"),
)
BY_KIND: dict[str, tuple[Lexeme, ...]] = {}
for lexeme in LEXICON:
    BY_KIND.setdefault(lexeme.kind, ()); BY_KIND[lexeme.kind] += (lexeme,)


class Grammar:
    def productions(self, symbol: Symbol) -> tuple[tuple[Symbol, ...], ...]:
        if symbol.name == "S":
            return ((Symbol("PP", "during", "event_pp"), Symbol("PP", "after", "event_pp"), Symbol("NP", "subject", "person"),
                     T("adv", "manner"), T("verb", "matrix_verb"), Symbol("CENTER", "center", "ordered_food"),
                     Symbol("PP", "for", "benefactive"), Symbol("PP", "near", "locative"),
                     Symbol("PP", "with", "accompaniment"), Symbol("PP", "in", "locative")),)
        if symbol.name == "CENTER":
            return ((T("inf", "center_inf"), T("verb", "center_verb"), T("adj_artifact", "center_adj"), T("noun_artifact", "center_noun")),)
        if symbol.name == "NP":
            return ((T("det", symbol.role + "_det"), T("adj_person", symbol.role + "_adj1"), T("adj_person", symbol.role + "_adj2"),
                     T("noun_person" if symbol.kind == "person" else "noun_org", symbol.role + "_noun")),)
        if symbol.name == "PP":
            noun = "noun_event" if symbol.kind == "event_pp" else ("noun_org" if symbol.role in {"near", "in"} else "noun_person")
            adj = "adj_event" if symbol.role in {"during", "after"} else "adj_person"
            if symbol.role in {"near", "in"}:
                return ((T("prep", symbol.role), T("det", symbol.role + "_det"), T(noun, symbol.role + "_noun")),)
            return ((T("prep", symbol.role), T("det", symbol.role + "_det"), T(adj, symbol.role + "_adj"), T(noun, symbol.role + "_noun")),)
        return ()

    def expand(self, symbol: Symbol) -> Node:
        rhs = self.productions(symbol)
        return Node(symbol, tuple(self.expand(child) for child in rhs[0])) if rhs else Node(symbol)

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


CENTER_KINDS = ("inf", "verb", "adj_artifact", "noun_artifact")


def live_bridge(left_words: tuple[str, str], right_words: tuple[str, str]) -> dict[str, object]:
    """Trace independently chosen leaves, pausing a leaf after each opening char."""
    leaves = {"left": [(left_words[1], left_words[1][::-1]), (left_words[0], left_words[0][::-1])],
              "right": [(right_words[0], right_words[0]), (right_words[1], right_words[1])]}
    cursor = {"left": [0, 0], "right": [0, 0]}; current = {"left": None, "right": None}
    residual = ""; owner = ""; preferred = "left"; trace: list[dict[str, object]] = []; cancellations = 0
    while True:
        for side in ("left", "right"):
            if current[side] is not None and cursor[side][1] >= len(current[side][1]): current[side] = None
        done = all(current[s] is None and cursor[s][0] >= len(leaves[s]) for s in ("left", "right"))
        if done: break
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
            "left_words": left_words, "right_words": right_words}


def semantically_licensed(center: tuple[Lexeme, ...]) -> bool:
    inf, verb, adjective, noun = center
    return (inf.meaning == "infinitive" and inf.word == "to" and verb.subject_type == "person" and
            verb.object_type == noun.meaning and adjective.kind == "adj_artifact" and noun.meaning == "food")


def discover_center(stats: Counter) -> dict[str, object] | None:
    channels = [BY_KIND[k] for k in CENTER_KINDS]
    for selected in itertools.product(*channels):
        stats["independent_center_assignments"] += 1
        if not semantically_licensed(selected): continue
        left, right = (selected[0].word, selected[1].word), (selected[2].word, selected[3].word)
        if any(a[::-1] == b for a in left for b in right): continue
        trace = live_bridge(left, right); stats["center_traces"] += 1
        if trace["completed"]:
            stats["center_complete_traces"] += 1
            return {"words": tuple(x.word for x in selected), "trace": trace,
                    "normalized": normalize_letters(" ".join(x.word for x in selected)),
                    "semantic_roles": [x.meaning for x in selected]}
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
    discovered = discover_center(Counter())
    if discovered is None or tuple(tokens[14:18]) != discovered["words"]: return False
    return assigned[11].meaning == "person" and assigned[13].subject_type == "person" and assigned[15].subject_type == "person"


def audit(grammar: Grammar, text: str, kind: str, provenance: tuple[str, ...]) -> dict[str, object]:
    exact = exact_audit(text); parsed = parse_tree(grammar, text); gate = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [key for key, value in gate.items() if not value]
    if not parsed: codes.append("independent_complete_reparse_failed")
    return {"record_kind": kind, "rendered": text, "provenance": provenance, "independent_exact_audit": exact,
            "independent_parse": parsed, "central_admission": gate, "mechanically_admitted": not codes,
            "rejection_codes": codes, "reader_status": "unreviewed; programmatic checks do not certify readability"}


def render(words: Iterable[str]) -> str:
    text = " ".join(words); return text[:1].upper() + text[1:] + "."


def run(*, state_limit: int = 100_000, closure_limit: int = 100) -> dict[str, object]:
    grammar = Grammar(); leaf_slots = slots(grammar.expand(Symbol("S"))); stats = Counter(state_count=0)
    center = discover_center(stats)
    if center is None: raise RuntimeError("independent typed channels found no complete center")
    stats["state_count"] += len(center["trace"]["trace"])
    centre = len(leaf_slots) // 2; words = list(center["words"])
    # This probe is an authored complete sentence tree around the discovered
    # centre; it is not a candidate and is independently reparsed below.
    prefix = "during the long trial after the detailed study the careful young chef deliberately decided".split()
    suffix = "for the patient editor near the archive with a diligent researcher in the museum".split()
    prefix2 = "after the long trial during the detailed study the careful young chef deliberately decided".split()
    controls = [render(prefix + words + suffix), render(prefix2 + words + suffix)]
    controls = [audit(grammar, text, "complete_connected_grammar_control", tuple(WORD.findall(text.lower()))) for text in controls]
    # Character frontier probe after centre completion.  It only emits one
    # character per state and stops at the first outer residual contradiction.
    assigned = {i: word for i, word in zip(range(centre - 2, centre + 2), words)}; lo, hi = centre - 3, centre + 2; residual = ""; owner = 0
    while (lo >= 0 or hi < len(leaf_slots)) and stats["state_count"] < state_limit:
        side = -owner if residual else (1 if lo >= 0 else -1); idx = lo if side == 1 else hi
        if (side == 1 and lo < 0) or (side == -1 and hi >= len(leaf_slots)): break
        opts = BY_KIND[leaf_slots[idx].symbol.kind]; item = next((x for x in opts if x.word not in assigned.values()), None)
        if item is None: break
        assigned[idx] = item.word; stats["outer_leaf_exposures"] += 1; stream = item.word[::-1] if side == 1 else item.word
        if side == 1: lo -= 1
        else: hi += 1
        for char in stream:
            stats["state_count"] += 1; stats["outer_char_emissions"] += 1
            if residual:
                if char != residual[0]: stats["outer_residual_contradictions"] += 1; break
                residual = residual[1:]; stats["outer_cancellations"] += 1
            else: residual = char; owner = side
        else: continue
        break
    return {"status": "independent_leaf_constraint_search_single_tree", "config": {"state_limit": state_limit, "closure_limit": closure_limit,
        "one_connected_tree": True, "grammar_owns_every_leaf": True, "independent_center_leaf_selection": True,
        "prebuilt_center_palindrome": False, "one_character_emission_states": True, "complete_reparse": True,
        "corpus_or_catalogue_generation": False}, "grammar_leaf_count": len(leaf_slots), "discovered_center": center,
        "stats": dict(stats), "exact_closures": [], "admitted_closures": [], "complete_grammar_controls": controls,
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(),
                       "material": "independent task-authored typed lexical channels; no prebuilt palindrome phrase, corpus, or catalogue"},
        "reader_facing_next_operator": "Replace the fixed outer lexical probe with a full typed frontier repair while preserving the independently discovered centre and its live residual schedule.",
        "scope": "No candidate is claimed; the discovered centre and controls are construction diagnostics, not human readability evidence."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--state-limit", type=int, default=100_000); parser.add_argument("--closure-limit", type=int, default=100); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit, closure_limit=args.closure_limit); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["state_count"], "exact": len(result["exact_closures"]), "admitted": len(result["admitted_closures"]), "center": result["discovered_center"]["words"]}, indent=2))


if __name__ == "__main__": main()
