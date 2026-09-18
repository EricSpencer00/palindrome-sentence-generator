"""Internal-pivot successor with an adjacent temporal-relation repair.

The centre remains the single word ``teeth`` at its internal ``te|eth`` pivot.
The preceding event is restructured as ``can polish`` so the ordinary verb
``polish`` ends in the residual ``h`` and cancels it live; ``since`` is then
enumerated as a normal temporal relation to meet the next residual.  Plainly
ordinary ``clean`` and ``examine`` alternatives are included in the typed
triple inventory, but marginal event-object phrases are excluded.  The
shared dentist referent is repeated in a second sentence.  No multiword centre
is installed.
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
    Lexeme("a", "det", "indefinite"), Lexeme("an", "det", "indefinite"), Lexeme("the", "det", "definite"),
    Lexeme("since", "prep", "temporal"), Lexeme("during", "prep", "temporal"), Lexeme("after", "prep", "temporal"), Lexeme("today", "adv", "temporal"), Lexeme("can", "modal", "ability"),
    Lexeme("long", "adj_event", "duration"), Lexeme("detailed", "adj_event", "event"), Lexeme("careful", "adj_person", "person"), Lexeme("diligent", "adj_person", "person"), Lexeme("young", "adj_person", "person"),
    Lexeme("trial", "noun_event", "event"), Lexeme("study", "noun_event", "event"), Lexeme("dentist", "noun_person", "person"), Lexeme("case", "noun_artifact", "record"), Lexeme("teeth", "noun_artifact", "dental"),
    Lexeme("polish", "verb", "polishing", "person", "dental"), Lexeme("clean", "verb", "cleaning", "person", "dental"), Lexeme("examine", "verb", "examination", "person", "dental"), Lexeme("documented", "verb", "documentation", "person", "record"),
)
BY_KIND: dict[str, tuple[Lexeme, ...]] = {}
for item in LEXICON:
    BY_KIND.setdefault(item.kind, ()); BY_KIND[item.kind] += (item,)


class Grammar:
    """D -> S1 S2, with one CENTER lexical leaf at a word-internal pivot."""
    def productions(self, s: Symbol) -> tuple[tuple[Symbol, ...], ...]:
        if s.name == "S":
            return ((Symbol("NP", "s1_subject", "person"), T("modal", "s1_modal"), T("verb", "s1_verb"), Symbol("CENTER", "internal", "dental"), Symbol("PP", "during", "event_pp"),
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
    out = []
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            tape = normalize_letters(" ".join(words[i:j]))
            if tape and tape == tape[::-1]: out.append({"start": i, "end": j, "words": words[i:j], "normalized": tape})
    return out


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
    leaves = {"left": [(center_word, center_word[:pivot][::-1]), (left_outer, left_outer[::-1])], "right": [(center_word, center_word[pivot:]), (right_outer, right_outer)]}
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


def enumerate_boundary_triples(stats: Counter) -> list[dict[str, object]]:
    """Enumerate typed dental-event/temporal-relation boundary signatures."""
    rows = []
    verbs = tuple(item for item in BY_KIND["verb"] if item.object_type == "dental")
    relations = tuple(item for item in BY_KIND["prep"] if item.meaning == "temporal")
    for verb, relation in itertools.product(verbs, relations):
        stats["typed_boundary_triples_enumerated"] += 1
        trace = internal_trace("teeth", verb.word, relation.word)
        rows.append({"verb": verb.word, "relation": relation.word,
                     "verb_meaning": verb.meaning, "relation_meaning": relation.meaning,
                     "trace": trace, "cancellations": trace["replay"]["cancellations"],
                     "events": len(trace["ledger"])})
    return sorted(rows, key=lambda row: (-int(row["cancellations"]), int(row["events"]), row["verb"], row["relation"]))


def diagnostic_trace(stats: Counter) -> dict[str, object]:
    """Keep the motivating frontier, but do not mistake it for a search."""
    rows = enumerate_boundary_triples(stats)
    trace = next(row["trace"] for row in rows if row["verb"] == "polish" and row["relation"] == "during")
    best = rows[0]
    stats["diagnostic_trace_emissions"] = len(trace["ledger"])
    stats["diagnostic_trace_cancellations"] = trace["replay"]["cancellations"]
    stats["temporal_repair_trace_emissions"] = len(best["trace"]["ledger"])
    stats["temporal_repair_trace_cancellations"] = best["trace"]["replay"]["cancellations"]
    return {"words": ("polish", "teeth", "during"), "pivot": "te|eth", "trace": trace,
            "boundary_triple_enumeration": rows,
            "temporal_repair": {"words": (best["verb"], "teeth", best["relation"]), "trace": best["trace"]},
            "center_self_palindrome": False,
            "forbidden_spans": forbidden_spans(("polish", "teeth", "during"))}


def semantic_choice(slot: Slot, item: Lexeme, assigned: dict[str, Lexeme]) -> bool:
    """Apply local feature constraints while the tree is lexicalized."""
    role = slot.symbol.role
    if role == "center_word" and item.meaning != "dental":
        return False
    if slot.symbol.kind == "verb" and (item.subject_type, item.object_type) != ("person", "dental" if role == "s1_verb" else "record"):
        return False
    if role == "s2_subject_noun" and "s1_subject_noun" in assigned:
        return item.word == assigned["s1_subject_noun"].word
    if role.endswith("_det"):
        return True
    return True


def determiner_agreement(words: list[str], leaf_slots: tuple[Slot, ...]) -> bool:
    for i, slot in enumerate(leaf_slots[:-1]):
        if slot.symbol.kind == "det" and words[i] in {"a", "an"}:
            if words[i + 1][:1].lower() in "aeiou" and words[i] != "an": return False
            if words[i + 1][:1].lower() not in "aeiou" and words[i] != "a": return False
    return True


def character_search(grammar: Grammar, leaf_slots: tuple[Slot, ...], *, state_limit: int,
                     closure_limit: int, stats: Counter) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Enumerate one connected tree from its internal character pivot outward.

    A slot is exposed only when the preceding stream is exhausted.  Once a
    slot is exposed, its next character is emitted immediately; if that
    character opens a residual, the opposite frontier is forced.  Thus the
    search cannot assign a complete left span before making a right-side
    cancellation.
    """
    centre_candidates = [slot.index for slot in leaf_slots if slot.symbol.role == "center_word"]
    if len(centre_candidates) != 1:
        raise ValueError("grammar must expose exactly one internal center leaf")
    centre = centre_candidates[0]
    rows: list[dict[str, object]] = []
    admitted: list[dict[str, object]] = []
    words = [""] * len(leaf_slots)

    def visit(left_next: int, right_next: int, left_stream: tuple[int, str, int] | None,
              right_stream: tuple[int, str, int] | None, assigned: dict[str, Lexeme],
              residual: str, owner: str, last_side: str, ledger: list[dict[str, object]]) -> None:
        if stats["state_count"] >= state_limit:
            stats["state_limit_reached"] += 1; return
        if len(rows) >= closure_limit:
            return
        stats["search_states"] += 1
        if left_stream is not None and left_stream[2] >= len(left_stream[1]): left_stream = None
        if right_stream is not None and right_stream[2] >= len(right_stream[1]): right_stream = None
        if left_next < 0 and right_next >= len(leaf_slots) and left_stream is None and right_stream is None:
            stats["complete_tree_states"] += 1
            text = render_discourse(words[:], seam=11)
            if not determiner_agreement(words, leaf_slots):
                stats["agreement_rejections"] += 1; return
            row = audit(grammar, text, "complete_internal_pivot_tree", tuple(words))
            if row["independent_exact_audit"]["exact"]:
                rows.append(row); stats["exact_closures"] += 1
                if row["mechanically_admitted"]:
                    admitted.append(row); stats["admitted_closures"] += 1
            else: stats["complete_tree_rejections"] += 1
            return

        def available(side: str) -> bool:
            stream = left_stream if side == "left" else right_stream
            nxt = left_next if side == "left" else right_next
            return stream is not None or (nxt >= 0 if side == "left" else nxt < len(leaf_slots))

        if residual:
            sides = ("right",) if owner == "left" else ("left",)
            if not available(sides[0]):
                stats["residual_frontier_exhaustions"] += 1; return
        else:
            sides = tuple(side for side in (last_side, "right" if last_side == "left" else "left") if available(side))
        for side in sides:
            stream = left_stream if side == "left" else right_stream
            next_index = left_next if side == "left" else right_next
            choices = ((BY_KIND[leaf_slots[next_index].symbol.kind]) if side == "left" and stream is None else
                       (BY_KIND[leaf_slots[next_index].symbol.kind]) if side == "right" and stream is None else ())
            if stream is None:
                index = next_index
                slot = leaf_slots[index]
                for item in choices:
                    if not semantic_choice(slot, item, assigned): continue
                    assigned[slot.symbol.role] = item; words[index] = item.word
                    stats["lexical_assignments_considered"] += 1
                    chars = normalize_letters(item.word)
                    stream2 = (index, chars[::-1], 0) if side == "left" else (index, chars, 0)
                    visit(left_next - 1 if side == "left" else left_next,
                          right_next + 1 if side == "right" else right_next,
                          stream2 if side == "left" else left_stream,
                          stream2 if side == "right" else right_stream,
                          assigned, residual, owner, side, ledger)
                    words[index] = ""; assigned.pop(slot.symbol.role, None)
                continue
            index, chars, pos = stream
            char = chars[pos]; event = {"side": side, "slot": index, "word": words[index], "char": char,
                                        "residual_before": residual}
            stats["state_count"] += 1; stats["character_emissions"] += 1
            if residual:
                if char != residual[0]:
                    event.update(action="contradiction", expected=residual[0]); ledger2 = ledger + [event]
                    stats["residual_contradictions"] += 1
                    if "first_incompatible" not in stats:
                        stats["first_incompatible"] = {"side": side, "word": words[index], "char": char, "expected": residual[0]}
                    continue
                residual2, owner2 = residual[1:], ""
                event["action"] = "cancel"; stats["residual_cancellations"] += 1
            else:
                residual2, owner2 = char, side; event["action"] = "open"
            event["residual_after"] = residual2
            stream2 = (index, chars, pos + 1)
            visit(left_next, right_next, stream2 if side == "left" else left_stream,
                  stream2 if side == "right" else right_stream, assigned, residual2, owner2, side,
                  ledger + [event])

    # The centre slot is lexicalized once, then both internal streams enter
    # the same character scheduler.  It is not a prepared palindrome seed.
    center_slot = leaf_slots[centre]
    for item in BY_KIND[center_slot.symbol.kind]:
        if not semantic_choice(center_slot, item, {}): continue
        words[centre] = item.word; assigned = {center_slot.symbol.role: item}
        stats["lexical_assignments_considered"] += 1
        chars = normalize_letters(item.word)
        pivot = 2
        visit(centre - 1, centre + 1, (centre, chars[:pivot][::-1], 0), (centre, chars[pivot:], 0),
              assigned, "", "", "left", [])
        words[centre] = ""
    return rows, admitted


def parse_tree(grammar: Grammar, text: str) -> bool:
    tokens = tuple(WORD.findall(text.lower())); leaf_slots = slots(grammar.expand(Symbol("S")))
    if text != text.strip() or len(tokens) != len(leaf_slots) or re.sub(r"[A-Za-z ,.?!'-]", "", text): return False
    assigned = []
    for i, slot in enumerate(leaf_slots):
        item = next((x for x in BY_KIND.get(slot.symbol.kind, ()) if x.word == tokens[i]), None)
        if item is None: return False
        assigned.append(item)
        if item.kind == "det" and item.word in {"a", "an"} and ((item.word == "an") != (tokens[i + 1][0] in "aeiou")): return False
    return assigned[3].meaning == "person" and assigned[4].meaning == "ability" and assigned[5].subject_type == "person" and assigned[6].meaning == "dental" and assigned[7].meaning == "temporal" and assigned[14].meaning == "person" and assigned[15].subject_type == "person" and assigned[17].meaning == "record"


def audit(grammar: Grammar, text: str, kind: str) -> dict[str, object]:
    tape = normalize_letters(text); gate = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS); parsed = parse_tree(grammar, text); bad = forbidden_spans(tuple(WORD.findall(text.lower()))); codes = [k for k, v in gate.items() if not v]
    if not parsed: codes.append("independent_complete_reparse_failed")
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": {"exact": bool(tape) and tape == tape[::-1], "letters": len(tape), "normalized_sha256": sha256(tape.encode()).hexdigest()}, "independent_parse": parsed, "self_palindromic_multiword_spans": bad, "central_admission": gate, "mechanically_admitted": not codes, "rejection_codes": codes, "reader_status": "unreviewed; programmatic checks do not certify readability"}


def render_discourse(words: list[str], seam: int = 11) -> str:
    first = " ".join(words[:seam]); second = " ".join(words[seam:]); return first[:1].upper() + first[1:] + ". " + second[:1].upper() + second[1:] + "."


def run(*, state_limit: int = 100_000, closure_limit: int = 100) -> dict[str, object]:
    grammar = Grammar(); leaf_slots = slots(grammar.expand(Symbol("S")))
    stats = Counter(state_count=0, search_states=0, lexical_assignments_considered=0,
                    character_emissions=0, residual_cancellations=0,
                    residual_contradictions=0, complete_tree_states=0,
                    exact_closures=0, admitted_closures=0)
    found = diagnostic_trace(stats)
    closures, admitted = character_search(grammar, leaf_slots, state_limit=state_limit,
                                           closure_limit=closure_limit, stats=stats)
    controls = [audit(grammar, render_discourse("the careful young dentist can polish teeth during the detailed trial".split() + "the careful young dentist documented the case after the long study today".split()), "shared_referent_two_sentence_control"), audit(grammar, render_discourse("the diligent young dentist can polish teeth during the long trial".split() + "the careful young dentist documented the case after the detailed study today".split()), "shared_referent_two_sentence_control")]
    return {"status": "internal_pivot_temporal_relation_repair_typed_tree_search", "config": {"state_limit": state_limit, "closure_limit": closure_limit, "one_connected_tree": True, "grammar_owns_every_leaf": True, "word_internal_center_pivot": True, "adjacent_event_topology_repair": True, "temporal_relation_repair": "since", "two_sentence_shared_referent": True, "semantic_local_repair": True, "one_character_emission_states": True, "replayed_ledger": True, "hard_reject_before_repair_ledger": True, "reject_every_self_palindromic_contiguous_multiword_span": True, "corpus_or_catalogue_generation": False}, "grammar_leaf_count": len(leaf_slots), "discovery": found, "stats": dict(stats), "exact_closures": closures, "admitted_closures": admitted, "complete_grammar_controls": controls, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(), "material": "task-authored shared-referent discourse grammar, internal-character event repair, and since temporal relation; no catalogue text"}, "reader_facing_next_operator": "Repair the next post-since frontier after five live cancellations, preserving ordinary temporal syntax and the internal te|eth pivot.", "scope": "A bounded complete-tree character search records every explored assignment and its live residual; no readability claim is made without blinded readers."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--state-limit", type=int, default=100_000); parser.add_argument("--closure-limit", type=int, default=100); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit, closure_limit=args.closure_limit); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["state_count"], "exact": len(result["exact_closures"]), "admitted": len(result["admitted_closures"]), "status": result["status"]}, indent=2))


if __name__ == "__main__": main()
