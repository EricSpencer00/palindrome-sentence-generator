"""Semantic local repair over a connected grammar with a replayed char ledger.

The generator chooses lexical leaves independently from typed channels.  A
character mismatch invokes a local, semantically licensed substitution and
replays the entire ledger from the frontier.  Every partial assignment is
checked for self-palindromic contiguous multiword spans; such assignments are
hard-rejected and never become candidates.
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
    word: str; kind: str; meaning: str; subject_type: str = ""; object_type: str = ""


@dataclass(frozen=True)
class Symbol:
    name: str; role: str = ""; kind: str = ""


@dataclass(frozen=True)
class Node:
    symbol: Symbol; children: tuple["Node", ...] = ()


@dataclass(frozen=True)
class Slot:
    index: int; symbol: Symbol


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
    def productions(self, s: Symbol) -> tuple[tuple[Symbol, ...], ...]:
        if s.name == "S":
            return ((Symbol("NP", "subject", "person"), Symbol("PP", "during", "event_pp"), Symbol("PP", "after", "event_pp"),
                     T("adv", "manner"), T("adv", "addition"), T("verb", "matrix_verb"), T("inf", "bridge_left"),
                     Symbol("CENTER", "center", "nonpal"), T("noun_artifact", "bridge_right"), Symbol("PP", "for", "benefactive"),
                     Symbol("PP", "near", "locative"), Symbol("PP", "with", "accompaniment"), Symbol("PP", "in", "locative"), T("adv", "today")),)
        if s.name == "CENTER": return ((T("verb", "center_verb"), T("adj_artifact", "center_adj")),)
        if s.name == "NP": return ((T("det", s.role + "_det"), T("adj_person", s.role + "_adj1"), T("adj_person", s.role + "_adj2"), T("noun_person", s.role + "_noun")),)
        if s.name == "PP":
            noun = "noun_event" if s.kind == "event_pp" else ("noun_org" if s.role in {"near", "in"} else "noun_person")
            adj = "adj_event" if s.role in {"during", "after"} else "adj_person"
            if s.role in {"near", "in"}: return ((T("prep", s.role), T("det", s.role + "_det"), T(noun, s.role + "_noun")),)
            return ((T("prep", s.role), T("det", s.role + "_det"), T(adj, s.role + "_adj"), T(noun, s.role + "_noun")),)
        return ()

    def expand(self, s: Symbol) -> Node:
        rhs = self.productions(s); return Node(s, tuple(self.expand(x) for x in rhs[0])) if rhs else Node(s)

    def digest(self) -> str:
        rows, seen, queue = [], set(), [Symbol("S")]
        while queue:
            s = queue.pop(0)
            if s in seen: continue
            seen.add(s); rhs = self.productions(s); rows.append((s.name, s.role, s.kind, [[(x.name, x.role, x.kind) for x in row] for row in rhs]))
            queue.extend(x for row in rhs for x in row if x.name != "T")
        return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def slots(tree: Node) -> tuple[Slot, ...]:
    out: list[Slot] = []
    def walk(n: Node) -> None:
        if n.symbol.name == "T": out.append(Slot(len(out), n.symbol))
        else:
            for child in n.children: walk(child)
    walk(tree); return tuple(out)


def spans(words: tuple[str, ...]) -> list[dict[str, object]]:
    out = []
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            tape = normalize_letters(" ".join(words[i:j]))
            if tape and tape == tape[::-1]: out.append({"start": i, "end": j, "words": words[i:j], "normalized": tape})
    return out


def replay(ledger: list[dict[str, object]]) -> dict[str, object]:
    residual = ""; cancellations = 0
    for event in ledger:
        char, side = str(event["char"]), str(event["side"])
        if residual:
            if char != residual[0]: return {"ok": False, "residual": residual, "cancellations": cancellations}
            residual = residual[1:]; cancellations += 1
        else: residual = char
        if side != event["side"]: return {"ok": False, "residual": residual, "cancellations": cancellations}
    return {"ok": True, "residual": residual, "cancellations": cancellations}


def live_window(left_center: str, right_center: str, left_outer: str, right_outer: str) -> dict[str, object]:
    leaves = {"left": [(left_center, left_center[::-1]), (left_outer, left_outer[::-1])], "right": [(right_center, right_center), (right_outer, right_outer)]}
    cur = {"left": None, "right": None}; pos = {"left": [0, 0], "right": [0, 0]}; residual = ""; owner = ""; preferred = "left"; ledger: list[dict[str, object]] = []
    while True:
        for side in ("left", "right"):
            if cur[side] is not None and pos[side][1] >= len(cur[side][1]): cur[side] = None
        if all(cur[s] is None and pos[s][0] >= len(leaves[s]) for s in ("left", "right")): break
        side = ("right" if owner == "left" else "left") if residual else ("left" if preferred == "left" else "right")
        if cur[side] is None:
            if pos[side][0] >= len(leaves[side]): return {"completed": False, "ledger": ledger, "first_incompatible": {"side": side, "reason": "frontier exhausted"}}
            idx = pos[side][0]; cur[side] = leaves[side][idx]; pos[side] = [idx + 1, 0]
        word, stream = cur[side]; char = stream[pos[side][1]]; pos[side][1] += 1; before = residual
        event = {"side": side, "word": word, "char": char, "residual_before": before}
        if residual and char != residual[0]:
            event["action"] = "contradiction"; ledger.append(event)
            return {"completed": False, "ledger": ledger, "first_incompatible": {"side": side, "word": word, "char": char, "expected": residual[0], "residual": before}}
        if residual: residual = residual[1:]; event["action"] = "cancel"
        else: residual = char; owner = side; event["action"] = "open"
        event["residual_after"] = residual; ledger.append(event)
        if not residual: preferred = "left"
    return {"completed": not residual, "ledger": ledger, "cancellations": sum(e["action"] == "cancel" for e in ledger), "replay": replay(ledger)}


def semantic_ok(verb: Lexeme, adjective: Lexeme, outer_left: Lexeme, outer_right: Lexeme) -> bool:
    return (verb.subject_type == "person" and verb.object_type == outer_right.meaning and adjective.meaning == "color" and
            outer_left.word == "to" and outer_right.meaning == "food")


def discover_with_local_repair(stats: Counter) -> dict[str, object]:
    channels = (BY_KIND["verb"], BY_KIND["adj_artifact"], BY_KIND["inf"], BY_KIND["noun_artifact"])
    initial = None; repaired = None
    for verb, adjective, outer_left, outer_right in itertools.product(*channels):
        stats["assignments"] += 1
        if not semantic_ok(verb, adjective, outer_left, outer_right): continue
        stats["semantic_assignments"] += 1
        words = (outer_left.word, verb.word, adjective.word, outer_right.word)
        if spans(words): stats["hard_span_rejections"] += 1; continue
        trace = live_window(verb.word, adjective.word, outer_left.word, outer_right.word); stats["initial_traces"] += 1
        initial = {"words": words, "trace": trace, "spans": spans(words)}
        if trace["completed"]: return {"initial": initial, "repair": None, "status": "complete_without_repair"}
        bad = trace.get("first_incompatible")
        stats["incompatible_frontiers"] += 1
        # Local repair: retain the exposed outer semantics, substitute only the
        # center verb, then replay the full ledger from the frontier.
        for replacement in BY_KIND["verb"]:
            if replacement.word == verb.word: continue
            if not semantic_ok(replacement, adjective, outer_left, outer_right): continue
            repaired_words = (outer_left.word, replacement.word, adjective.word, outer_right.word)
            repaired_trace = live_window(replacement.word, adjective.word, outer_left.word, outer_right.word); stats["repair_replays"] += 1
            repaired = {"operator": "semantic_center_verb_substitution", "from_words": words, "to_words": repaired_words,
                        "trigger": bad, "trace": repaired_trace, "spans": spans(repaired_words),
                        "hard_span_reject": bool(spans(repaired_words)), "ledger_replay_match": repaired_trace.get("replay", {}).get("ok")}
            if repaired_trace["completed"]: stats["repair_complete_traces"] += 1
            return {"initial": initial, "repair": repaired, "status": "repaired_but_hard_span_rejected" if repaired["hard_span_reject"] else "repaired"}
    raise RuntimeError("no semantically licensed local repair path")


def parse_tree(grammar: Grammar, text: str) -> bool:
    if text != text.strip() or re.sub(r"[A-Za-z ,.?!'-]", "", text): return False
    tokens = tuple(WORD.findall(text.lower())); leaf_slots = slots(grammar.expand(Symbol("S")))
    if len(tokens) != len(leaf_slots): return False
    assigned = []
    for i, slot in enumerate(leaf_slots):
        item = next((x for x in BY_KIND.get(slot.symbol.kind, ()) if x.word == tokens[i]), None)
        if item is None: return False
        assigned.append(item)
        if item.kind == "det" and item.word in {"a", "an"} and ((item.word == "an") != (tokens[i + 1][0] in "aeiou")): return False
    return assigned[3].meaning == "person" and assigned[14].subject_type == "person" and assigned[16].subject_type == "person" and assigned[17].meaning == "color" and assigned[18].meaning == "food"


def audit(grammar: Grammar, text: str, kind: str) -> dict[str, object]:
    exact = normalize_letters(text); gate = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS); parsed = parse_tree(grammar, text); bad_spans = spans(tuple(WORD.findall(text.lower())))
    codes = [k for k, v in gate.items() if not v]
    if bad_spans: codes.append("self_palindromic_multiword_span")
    if not parsed: codes.append("independent_complete_reparse_failed")
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": {"exact": exact == exact[::-1], "letters": len(exact), "normalized_sha256": sha256(exact.encode()).hexdigest()}, "independent_parse": parsed, "self_palindromic_multiword_spans": bad_spans, "central_admission": gate, "mechanically_admitted": not codes, "rejection_codes": codes, "reader_status": "unreviewed; programmatic checks do not certify readability"}


def render(words: Iterable[str]) -> str:
    text = " ".join(words); return text[:1].upper() + text[1:] + "."


def run(*, state_limit: int = 100_000, closure_limit: int = 100) -> dict[str, object]:
    grammar = Grammar(); leaf_slots = slots(grammar.expand(Symbol("S"))); stats = Counter(state_count=0); found = discover_with_local_repair(stats); stats["state_count"] += len(found["initial"]["trace"]["ledger"]) + (len(found["repair"]["trace"]["ledger"]) if found["repair"] else 0)
    discovered = found["repair"]["to_words"] if found["repair"] else found["initial"]["words"]
    prefix = "the careful young chef during the long trial after the detailed study deliberately also decided".split() + ["to"]
    suffix = "for the patient editor near the archive with a diligent researcher in the museum today".split()
    controls = [audit(grammar, render(prefix + list(discovered[1:]) + suffix), "complete_connected_grammar_control"), audit(grammar, render("the patient diligent chef during the detailed trial after the long study deliberately also decided".split() + ["to"] + list(discovered[1:]) + suffix), "complete_connected_grammar_control")]
    return {"status": "semantic_local_repair_frontier_single_tree", "config": {"state_limit": state_limit, "closure_limit": closure_limit, "one_connected_tree": True, "grammar_owns_every_leaf": True, "semantic_local_repair": True, "one_character_emission_states": True, "replayed_ledger": True, "reject_every_self_palindromic_contiguous_multiword_span": True, "corpus_or_catalogue_generation": False}, "grammar_leaf_count": len(leaf_slots), "discovery": found, "stats": dict(stats), "exact_closures": [], "admitted_closures": [], "complete_grammar_controls": controls, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(), "material": "task-authored connected grammar, semantic roles, and local repair alternatives; no corpus/catalogue text"}, "reader_facing_next_operator": "Repair the first incompatible outer frontier with a typed multiword constituent substitution while retaining ledger replay and the hard span rejection.", "scope": "All exact-looking local traces containing a forbidden self-palindromic multiword span are rejected; no candidate or readability claim is made."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--state-limit", type=int, default=100_000); parser.add_argument("--closure-limit", type=int, default=100); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit, closure_limit=args.closure_limit); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["state_count"], "exact": 0, "admitted": 0, "status": result["status"]}, indent=2))


if __name__ == "__main__": main()
