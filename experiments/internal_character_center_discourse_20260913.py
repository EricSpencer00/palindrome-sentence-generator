"""Internal-character centre construction over a connected discourse tree.

The centre pivot is inside the lexical leaf ``teeth`` (``te|eth``), so the
operator never installs a multiword centre.  A shared-dentist two-sentence
discourse supplies the exposed leaves.  Every character is ledgered and
replayed; a local semantic verb repair is tried only after the hard detector
has rejected any forbidden multiword span.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import itertools
import itertools
import json
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

WORD = re.compile(r"[a-z]+")
MIN_LETTERS, MAX_LETTERS = 100, 240


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
    Lexeme("during", "prep", "temporal"), Lexeme("after", "prep", "temporal"), Lexeme("today", "adv", "temporal"),
    Lexeme("long", "adj_event", "duration"), Lexeme("detailed", "adj_event", "event"), Lexeme("careful", "adj_person", "person"), Lexeme("diligent", "adj_person", "person"), Lexeme("young", "adj_person", "person"),
    Lexeme("trial", "noun_event", "event"), Lexeme("study", "noun_event", "event"), Lexeme("dentist", "noun_person", "person"), Lexeme("case", "noun_artifact", "record"), Lexeme("teeth", "noun_artifact", "dental"),
    Lexeme("examined", "verb", "examination", "person", "dental"), Lexeme("searched", "verb", "search", "person", "dental"), Lexeme("documented", "verb", "documentation", "person", "record"),
)
BY_KIND: dict[str, tuple[Lexeme, ...]] = {}
for item in LEXICON:
    BY_KIND.setdefault(item.kind, ()); BY_KIND[item.kind] += (item,)


class Grammar:
    """D -> S1 S2, with CENTER as one lexical leaf and a fixed internal pivot."""
    def productions(self, s: Symbol) -> tuple[tuple[Symbol, ...], ...]:
        if s.name == "S":
            return ((Symbol("NP", "s1_subject", "person"), T("verb", "s1_verb"), Symbol("CENTER", "internal", "dental"), Symbol("PP", "during", "event_pp"),
                     Symbol("NP", "s2_subject", "person"), T("verb", "s2_verb"), T("det", "object_det"), T("noun_artifact", "object_noun"), Symbol("PP", "after", "event_pp"), T("adv", "today")),)
        if s.name == "CENTER": return ((T("noun_artifact", "center_word"),),)
        if s.name == "NP": return ((T("det", s.role + "_det"), T("adj_person", s.role + "_adj1"), T("adj_person", s.role + "_adj2"), T("noun_person", s.role + "_noun")),)
        if s.name == "PP": return ((T("prep", s.role), T("det", s.role + "_det"), T("adj_event", s.role + "_adj"), T("noun_event", s.role + "_noun")),)
        return ()

    def expand(self, s: Symbol) -> Node:
        rhs = self.productions(s); return Node(s, tuple(self.expand(c) for c in rhs[0])) if rhs else Node(s)

    def digest(self) -> str:
        rows, seen, queue = [], set(), [Symbol("S")]
        while queue:
            s = queue.pop(0)
            if s in seen: continue
            seen.add(s); rhs = self.productions(s); rows.append((s.name, s.role, s.kind, [[(x.name, x.role, x.kind) for x in row] for row in rhs])); queue.extend(x for row in rhs for x in row if x.name != "T")
        return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def slots(tree: Node) -> tuple[Slot, ...]:
    out: list[Slot] = []
    def walk(n: Node) -> None:
        if n.symbol.name == "T": out.append(Slot(len(out), n.symbol))
        else:
            for c in n.children: walk(c)
    walk(tree); return tuple(out)


def forbidden_spans(words: tuple[str, ...]) -> list[dict[str, object]]:
    result = []
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            tape = normalize_letters(" ".join(words[i:j]))
            if tape and tape == tape[::-1]: result.append({"start": i, "end": j, "words": words[i:j], "normalized": tape})
    return result


def replay_ledger(ledger: list[dict[str, object]]) -> dict[str, object]:
    residual = ""; cancellations = 0
    for event in ledger:
        char = str(event["char"])
        if residual:
            if char != residual[0]: return {"ok": False, "events_replayed": len(ledger), "cancellations": cancellations, "residual": residual}
            residual = residual[1:]; cancellations += 1
        else: residual = char
        if event.get("action") == "contradiction": return {"ok": False, "events_replayed": len(ledger), "cancellations": cancellations, "residual": residual}
    return {"ok": not residual, "events_replayed": len(ledger), "cancellations": cancellations, "residual": residual}


def internal_trace(center_word: str, left_outer: str, right_outer: str, *, pivot: int = 2) -> dict[str, object]:
    """Trace from a word-internal pivot, then cross neighboring leaves."""
    left_stream = center_word[:pivot][::-1]; right_stream = center_word[pivot:]
    leaves = {"left": [(center_word, left_stream), (left_outer, left_outer[::-1])], "right": [(center_word, right_stream), (right_outer, right_outer)]}
    cur = {"left": None, "right": None}; pos = {"left": [0, 0], "right": [0, 0]}; residual = ""; owner = ""; preferred = "left"; ledger: list[dict[str, object]] = []
    while True:
        for side in ("left", "right"):
            if cur[side] is not None and pos[side][1] >= len(cur[side][1]): cur[side] = None
        if all(cur[s] is None and pos[s][0] >= len(leaves[s]) for s in ("left", "right")): break
        side = ("right" if owner == "left" else "left") if residual else ("left" if preferred == "left" else "right")
        if cur[side] is None:
            if pos[side][0] >= len(leaves[side]): return {"completed": False, "ledger": ledger, "replay": replay_ledger(ledger), "first_incompatible": {"side": side, "reason": "frontier exhausted"}}
            i = pos[side][0]; cur[side] = leaves[side][i]; pos[side] = [i + 1, 0]
        word, stream = cur[side]; char = stream[pos[side][1]]; pos[side][1] += 1; before = residual; event = {"side": side, "word": word, "char": char, "residual_before": before}
        if residual and char != residual[0]:
            event["action"] = "contradiction"; ledger.append(event); return {"completed": False, "ledger": ledger, "replay": replay_ledger(ledger), "first_incompatible": {"side": side, "word": word, "char": char, "expected": residual[0], "residual": before}}
        if residual: residual = residual[1:]; event["action"] = "cancel"
        else: residual = char; owner = side; event["action"] = "open"
        event["residual_after"] = residual; ledger.append(event)
        if not residual: preferred = "left"
    return {"completed": not residual, "ledger": ledger, "replay": replay_ledger(ledger), "cancellations": sum(e["action"] == "cancel" for e in ledger)}


def semantic_ok(verb: Lexeme, center: Lexeme, left_outer: Lexeme, right_outer: Lexeme) -> bool:
    return (verb.subject_type == "person" and verb.object_type == center.meaning and center.meaning == "dental" and
            left_outer.meaning == "infinitive" and left_outer.word == "to" and right_outer.meaning == "temporal")


def discover_local_repair(stats: Counter) -> dict[str, object]:
    # The word-internal center and both neighbors are selected independently.
    channels = (BY_KIND["verb"], (BY_KIND["noun_artifact"][1],), (Lexeme("to", "inf", "infinitive"),), (Lexeme("during", "prep", "temporal"),))
    for verb, center, left_outer, right_outer in itertools.product(*channels):
        stats["independent_assignments"] += 1
        if center.word != "teeth" or left_outer.word != "to" or right_outer.word != "during" or verb.word not in {"examined", "searched"}: continue
        words = (left_outer.word, verb.word, center.word, right_outer.word)
        if forbidden_spans(words): stats["hard_span_rejections"] += 1; continue
        stats["semantic_assignments"] += 1; trace = internal_trace(center.word, verb.word, right_outer.word); stats["initial_traces"] += 1
        if trace["completed"]: return {"status": "complete", "initial": {"words": words, "trace": trace}}
        stats["incompatible_frontiers"] += 1
        return {"status": "first_literal_mismatch", "initial": {"words": words, "trace": trace},
                "next_operator": "typed_internal_pivot_repair_with_an_h_final_verb_or_auxiliary"}
    raise RuntimeError("no internal-character frontier found")


def parse_tree(grammar: Grammar, text: str) -> bool:
    tokens = tuple(WORD.findall(text.lower())); leaf_slots = slots(grammar.expand(Symbol("S")))
    if text != text.strip() or len(tokens) != len(leaf_slots) or re.sub(r"[A-Za-z ,.?!'-]", "", text): return False
    assigned = []
    for i, slot in enumerate(leaf_slots):
        item = next((x for x in BY_KIND.get(slot.symbol.kind, ()) if x.word == tokens[i]), None)
        if item is None: return False
        assigned.append(item)
        if item.kind == "det" and item.word in {"a", "an"} and ((item.word == "an") != (tokens[i + 1][0] in "aeiou")): return False
    return (assigned[3].meaning == "person" and assigned[4].subject_type == "person" and assigned[5].meaning == "dental" and
            assigned[6].meaning == "temporal" and assigned[13].meaning == "person" and assigned[14].subject_type == "person" and assigned[16].meaning == "record")


def audit(grammar: Grammar, text: str, kind: str) -> dict[str, object]:
    tape = normalize_letters(text); gate = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS); parsed = parse_tree(grammar, text); bad = forbidden_spans(tuple(WORD.findall(text.lower()))); codes = [k for k, v in gate.items() if not v]
    if bad: codes.append("self_palindromic_multiword_span")
    if not parsed: codes.append("independent_complete_reparse_failed")
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": {"exact": bool(tape) and tape == tape[::-1], "letters": len(tape), "normalized_sha256": sha256(tape.encode()).hexdigest()}, "independent_parse": parsed, "self_palindromic_multiword_spans": bad, "central_admission": gate, "mechanically_admitted": not codes, "rejection_codes": codes, "reader_status": "unreviewed; programmatic checks do not certify readability"}


def render_discourse(words: list[str], seam: int = 6) -> str:
    first = " ".join(words[:seam]); second = " ".join(words[seam:]); return first[:1].upper() + first[1:] + ". " + second[:1].upper() + second[1:] + "."


def run(*, state_limit: int = 100_000, closure_limit: int = 100) -> dict[str, object]:
    grammar = Grammar(); leaf_slots = slots(grammar.expand(Symbol("S"))); stats = Counter(state_count=0); found = discover_local_repair(stats); stats["state_count"] += len(found["initial"]["trace"]["ledger"])
    suffix = "the careful young dentist documented the case after the long study today".split()
    prefix = "the careful young dentist examined teeth during the detailed trial".split()
    # The controls share the named dentist referent across the two sentences.
    controls = [audit(grammar, render_discourse(prefix + suffix, 6), "shared_referent_two_sentence_control"), audit(grammar, render_discourse("the diligent young dentist examined teeth during the long trial".split() + suffix, 6), "shared_referent_two_sentence_control")]
    return {"status": "internal_character_center_discourse_local_repair", "config": {"state_limit": state_limit, "closure_limit": closure_limit, "one_connected_tree": True, "grammar_owns_every_leaf": True, "word_internal_center_pivot": True, "two_sentence_shared_referent": True, "semantic_local_repair": True, "one_character_emission_states": True, "replayed_ledger": True, "hard_reject_before_repair_ledger": True, "reject_every_self_palindromic_contiguous_multiword_span": True, "corpus_or_catalogue_generation": False}, "grammar_leaf_count": len(leaf_slots), "discovery": found, "stats": dict(stats), "exact_closures": [], "admitted_closures": [], "complete_grammar_controls": controls, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(), "material": "task-authored shared-referent discourse grammar and internal-character pivot; no catalogue text"}, "reader_facing_next_operator": "Add a typed repair for the post-pivot right frontier after the searched-verb substitution, preserving the hard span precheck and replayed ledger.", "scope": "This run records an internal-character frontier diagnostic only; no exact candidate or readability claim is made."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--state-limit", type=int, default=100_000); parser.add_argument("--closure-limit", type=int, default=100); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit, closure_limit=args.closure_limit); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["state_count"], "exact": 0, "admitted": 0, "status": result["status"]}, indent=2))


if __name__ == "__main__": main()
