"""Two-sentence discourse grammar with shifted boundary local repair.

The first sentence ends in an infinitive and the second begins with a noun
phrase.  Their live character residual crosses the sentence boundary, but no
fixed palindrome phrase is installed.  A hard detector rejects every
self-palindromic contiguous span of two or more words *before* a local repair
is proposed.  The ledger is replayed independently for every trace.
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
    Lexeme("a", "det", "indefinite"), Lexeme("an", "det", "indefinite"), Lexeme("the", "det", "definite"), Lexeme("no", "det_no", "negative"),
    Lexeme("during", "prep", "temporal"), Lexeme("after", "prep", "temporal"), Lexeme("near", "prep", "locative"), Lexeme("with", "prep", "accompaniment"), Lexeme("in", "prep", "locative"),
    Lexeme("long", "adj_event", "duration"), Lexeme("detailed", "adj_event", "event"), Lexeme("careful", "adj_person", "person"), Lexeme("young", "adj_person", "person"), Lexeme("patient", "adj_person", "person"), Lexeme("diligent", "adj_person", "person"),
    Lexeme("trial", "noun_event", "event"), Lexeme("study", "noun_event", "event"), Lexeme("tailor", "noun_person", "person"), Lexeme("editor", "noun_person", "person"), Lexeme("researcher", "noun_person", "person"), Lexeme("archive", "noun_org", "organization"), Lexeme("museum", "noun_org", "organization"),
    Lexeme("riot", "noun_event", "event"), Lexeme("strike", "noun_event", "event"), Lexeme("to", "inf", "infinitive"), Lexeme("deliberately", "adv", "manner"), Lexeme("today", "adv", "temporal"),
    Lexeme("came", "verb", "motion", "person", "event"), Lexeme("followed", "verb", "following", "event", "person"), Lexeme("paint", "verb", "painting", "person", "artifact"), Lexeme("iron", "verb", "ironing", "person", "artifact"),
)
BY_KIND: dict[str, tuple[Lexeme, ...]] = {}
for item in LEXICON:
    BY_KIND.setdefault(item.kind, ()); BY_KIND[item.kind] += (item,)


class Grammar:
    """D -> S1 S2; CENTER is a single connected node across the discourse seam."""
    def productions(self, s: Symbol) -> tuple[tuple[Symbol, ...], ...]:
        if s.name == "S":
            return ((Symbol("NP", "subject", "person"), Symbol("PP", "during", "event_pp"), Symbol("PP", "after", "event_pp"), T("adv", "manner"), T("verb", "matrix_verb"), T("inf", "bridge_left"), Symbol("CENTER", "center", "discourse_seam"), T("verb", "followed"), Symbol("NP", "object", "person"), Symbol("PP", "near", "locative"), Symbol("PP", "with", "accompaniment"), Symbol("PP", "in", "locative"), T("adv", "today")),)
        if s.name == "CENTER": return ((T("verb", "center_verb"), T("det_no", "center_det"), T("noun_event", "center_noun")),)
        if s.name == "NP": return ((T("det", s.role + "_det"), T("adj_person", s.role + "_adj1"), T("adj_person", s.role + "_adj2"), T("noun_person", s.role + "_noun")),)
        if s.name == "PP":
            noun = "noun_event" if s.kind == "event_pp" else ("noun_org" if s.role in {"near", "in"} else "noun_person")
            adj = "adj_event" if s.role in {"during", "after"} else "adj_person"
            if s.role in {"near", "in"}: return ((T("prep", s.role), T("det", s.role + "_det"), T(noun, s.role + "_noun")),)
            return ((T("prep", s.role), T("det", s.role + "_det"), T(adj, s.role + "_adj"), T(noun, s.role + "_noun")),)
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
    out = []
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            tape = normalize_letters(" ".join(words[i:j]))
            if tape and tape == tape[::-1]: out.append({"start": i, "end": j, "words": words[i:j], "normalized": tape})
    return out


def replay_ledger(ledger: list[dict[str, object]]) -> dict[str, object]:
    residual = ""; cancellations = 0; prefix_ok = True
    for event in ledger:
        char, side = str(event["char"]), str(event["side"])
        if residual:
            if char != residual[0]: prefix_ok = False; return {"ok": False, "prefix_ok": prefix_ok, "residual": residual, "cancellations": cancellations, "events_replayed": len(ledger)}
            residual = residual[1:]; cancellations += 1
        else: residual = char
        if event.get("action") == "contradiction": prefix_ok = False; return {"ok": False, "prefix_ok": prefix_ok, "residual": residual, "cancellations": cancellations, "events_replayed": len(ledger)}
    return {"ok": not residual, "prefix_ok": prefix_ok, "residual": residual, "cancellations": cancellations, "events_replayed": len(ledger)}


def live_seam(left_inner: str, right_det: str, right_noun: str, left_outer: str) -> dict[str, object]:
    leaves = {"left": [(left_inner, left_inner[::-1]), (left_outer, left_outer[::-1])], "right": [(right_det, right_det), (right_noun, right_noun)]}
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


def semantic_ok(verb: Lexeme, det: Lexeme, noun: Lexeme, outer_left: Lexeme, outer_right: Lexeme) -> bool:
    return verb.subject_type == "person" and verb.object_type == "artifact" and det.word == "no" and noun.meaning == "event" and outer_left.meaning == "infinitive" and outer_right.meaning == "event"


def discover_repair(stats: Counter) -> dict[str, object]:
    channels = (BY_KIND["verb"], BY_KIND["det_no"], BY_KIND["noun_event"], BY_KIND["inf"], BY_KIND["noun_event"])
    for verb, det, noun, outer_left, outer_right in itertools.product(*channels):
        stats["independent_assignments"] += 1
        if not semantic_ok(verb, det, noun, outer_left, outer_right): continue
        words = (outer_left.word, verb.word, det.word, noun.word)
        if forbidden_spans(words): stats["hard_repair_pre_rejections"] += 1; continue
        stats["semantic_assignments"] += 1
        initial = live_seam(verb.word, det.word, noun.word, outer_left.word); stats["initial_traces"] += 1
        if initial["completed"]: return {"status": "complete_without_repair", "initial": {"words": words, "trace": initial}}
        stats["incompatible_frontiers"] += 1
        for replacement in BY_KIND["verb"]:
            if replacement.word == verb.word or not semantic_ok(replacement, det, noun, outer_left, outer_right): continue
            repaired_words = (outer_left.word, replacement.word, det.word, noun.word)
            # The hard span detector runs before this proposed repair is
            # ledger-eligible; rejected span evidence is retained explicitly.
            bad = forbidden_spans(repaired_words)
            if bad:
                stats["repair_blocked_before_ledger"] += 1
                return {"status": "repair_blocked_by_hard_span", "initial": {"words": words, "trace": initial}, "repair": {"operator": "semantic_center_verb_substitution", "from_words": words, "to_words": repaired_words, "blocked_before_ledger": True, "forbidden_spans": bad, "trace": None}}
    raise RuntimeError("no incompatible semantic frontier found")


def parse_tree(grammar: Grammar, text: str) -> bool:
    tokens = tuple(WORD.findall(text.lower())); leaf_slots = slots(grammar.expand(Symbol("S")))
    if text != text.strip() or len(tokens) != len(leaf_slots) or re.sub(r"[A-Za-z ,.?!'-]", "", text): return False
    assigned = []
    for i, slot in enumerate(leaf_slots):
        item = next((x for x in BY_KIND.get(slot.symbol.kind, ()) if x.word == tokens[i]), None)
        if item is None: return False
        assigned.append(item)
        if item.kind == "det" and item.word in {"a", "an"} and ((item.word == "an") != (tokens[i + 1][0] in "aeiou")): return False
    return (assigned[3].meaning == "person" and assigned[13].subject_type == "person" and
            assigned[14].meaning == "infinitive" and assigned[15].subject_type == "person" and
            assigned[16].meaning == "negative" and assigned[17].meaning == "event")


def audit(grammar: Grammar, text: str, kind: str) -> dict[str, object]:
    tape = normalize_letters(text); gate = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS); parsed = parse_tree(grammar, text); bad = forbidden_spans(tuple(WORD.findall(text.lower())))
    codes = [k for k, v in gate.items() if not v]
    if bad: codes.append("self_palindromic_multiword_span")
    if not parsed: codes.append("independent_complete_reparse_failed")
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": {"exact": bool(tape) and tape == tape[::-1], "letters": len(tape), "normalized_sha256": sha256(tape.encode()).hexdigest()}, "independent_parse": parsed, "self_palindromic_multiword_spans": bad, "central_admission": gate, "mechanically_admitted": not codes, "rejection_codes": codes, "reader_status": "unreviewed; programmatic checks do not certify readability"}


def render(words: Iterable[str]) -> str:
    text = " ".join(words); return text[:1].upper() + text[1:] + "."


def render_discourse(words: list[str], seam: int = 16) -> str:
    """Render the grammar's two connected sentences at the shifted seam."""
    first = " ".join(words[:seam]); second = " ".join(words[seam:])
    return first[:1].upper() + first[1:] + ". " + second[:1].upper() + second[1:] + "."


def run(*, state_limit: int = 100_000, closure_limit: int = 100) -> dict[str, object]:
    grammar = Grammar(); leaf_slots = slots(grammar.expand(Symbol("S"))); stats = Counter(state_count=0); found = discover_repair(stats); stats["state_count"] += len(found["initial"]["trace"]["ledger"])
    # An intact two-sentence discourse control with independently checked
    # agreement/valency; punctuation is only sentence punctuation.
    prefix = "the careful young tailor during the long trial after the detailed study deliberately came to".split()
    center = found["initial"]["words"]
    suffix = "followed the patient diligent editor near the archive with a diligent researcher in the museum today".split()
    control = render(prefix + list(center[1:]) + suffix)
    alt = render("the patient careful tailor during the detailed trial after the long study deliberately came to".split() + list(center[1:]) + suffix)
    controls = [audit(grammar, render_discourse(prefix + list(center[1:]) + suffix), "complete_connected_discourse_control"), audit(grammar, render_discourse("the patient careful tailor during the detailed trial after the long study deliberately came to".split() + list(center[1:]) + suffix), "complete_connected_discourse_control")]
    return {"status": "semantic_local_repair_frontier_two_sentence", "config": {"state_limit": state_limit, "closure_limit": closure_limit, "one_connected_tree": True, "grammar_owns_every_leaf": True, "two_sentence_discourse_topology": True, "semantic_local_repair": True, "one_character_emission_states": True, "replayed_ledger": True, "hard_reject_before_repair_ledger": True, "reject_every_self_palindromic_contiguous_multiword_span": True, "corpus_or_catalogue_generation": False}, "grammar_leaf_count": len(leaf_slots), "discovery": found, "stats": dict(stats), "exact_closures": [], "admitted_closures": [], "complete_grammar_controls": controls, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(), "material": "task-authored two-sentence grammar, semantic alternatives, and local replay ledger; no catalogue text"}, "reader_facing_next_operator": "Use a semantic multiword constituent repair at the first discourse seam whose proposed replacement survives the hard-span precheck, then replay the complete ledger before any admission attempt.", "scope": "The incompatible frontier and pre-ledger hard rejection are construction evidence only; no candidate or readability claim is made."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--state-limit", type=int, default=100_000); parser.add_argument("--closure-limit", type=int, default=100); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit, closure_limit=args.closure_limit); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["state_count"], "exact": 0, "admitted": 0, "status": result["status"]}, indent=2))


if __name__ == "__main__": main()
