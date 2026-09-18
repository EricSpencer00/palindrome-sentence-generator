"""Fresh ordinary internal-centre inventory and connected-tree search.

This retires the exhausted teeth and career frames.  It enumerates ordinary single-word
centres (attack, effect) at every internal pivot, filters them through
frame-level lexical semantics, then runs the same one-character frontier
scheduler over one complete discourse tree.  No multiword centre is exposed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import itertools
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

WORD = re.compile(r"[a-z]+")
MIN_LETTERS, MAX_LETTERS = 100, 260


class Lexeme:
    def __init__(self, word: str, kind: str, meaning: str, subject_type: str = "", object_type: str = ""):
        self.word, self.kind, self.meaning = word, kind, meaning
        self.subject_type, self.object_type = subject_type, object_type


class Symbol:
    def __init__(self, name: str, role: str = "", kind: str = ""):
        self.name, self.role, self.kind = name, role, kind
    def __hash__(self): return hash((self.name, self.role, self.kind))
    def __eq__(self, other): return isinstance(other, Symbol) and (self.name, self.role, self.kind) == (other.name, other.role, other.kind)


class Node:
    def __init__(self, symbol: Symbol, children: tuple["Node", ...] = ()):
        self.symbol, self.children = symbol, children


class Slot:
    def __init__(self, index: int, symbol: Symbol): self.index, self.symbol = index, symbol


def T(kind: str, role: str) -> Symbol: return Symbol("T", role, kind)


LEXICON = (
    Lexeme("a", "det", "indefinite"), Lexeme("an", "det", "indefinite"), Lexeme("the", "det", "definite"),
    Lexeme("at", "prep", "institution_event"), Lexeme("during", "prep", "temporal"), Lexeme("after", "prep", "temporal"),
    Lexeme("today", "adv", "temporal"), Lexeme("annual", "adj_event", "event"), Lexeme("public", "adj_event", "event"), Lexeme("western", "adj_event", "event"),
    Lexeme("careful", "adj_person", "person"), Lexeme("patient", "adj_person", "person"), Lexeme("sudden", "adj_center", "attack"), Lexeme("public", "adj_center", "attack"), Lexeme("visible", "adj_center", "effect"),
    Lexeme("researcher", "noun_person", "person"), Lexeme("analyst", "noun_person", "person"),
    Lexeme("career", "noun_center", "career"), Lexeme("attack", "noun_center", "attack"), Lexeme("effect", "noun_center", "effect"),
    Lexeme("institute", "noun_event", "institution"), Lexeme("clinic", "noun_event", "institution"), Lexeme("trial", "noun_event", "event"), Lexeme("study", "noun_event", "event"),
    Lexeme("report", "noun_artifact", "record"), Lexeme("record", "noun_artifact", "record"),
    Lexeme("began", "verb_intro", "beginning", "person", "career"), Lexeme("described", "verb_intro", "description", "person", "attack"), Lexeme("measured", "verb_intro", "measurement", "person", "effect"),
    Lexeme("documented", "verb_record", "documentation", "person", "record"), Lexeme("reviewed", "verb_record", "review", "person", "record"),
)
BY_KIND: dict[str, tuple[Lexeme, ...]] = {}
for item in LEXICON: BY_KIND[item.kind] = BY_KIND.get(item.kind, ()) + (item,)

FRAME = {
    "attack": {"verb": "described", "adjectives": {"sudden", "public"}, "relations": {"during"}},
    "effect": {"verb": "measured", "adjectives": {"visible"}, "relations": {"during"}},
}
PIVOTS = {"attack": (2,), "effect": (2,)}


class Grammar:
    def productions(self, s: Symbol) -> tuple[tuple[Symbol, ...], ...]:
        if s.name == "S":
            return ((Symbol("NP", "s1_subject", "person"), T("verb_intro", "intro_verb"), T("det", "center_det"), T("adj_center", "center_adj"), T("noun_center", "center_word"), Symbol("PP", "relation", "event_pp"), Symbol("NP", "s2_subject", "person"), T("verb_record", "record_verb"), T("det", "object_det"), T("adj_event", "object_adj"), T("noun_artifact", "object_noun"), Symbol("PP", "after", "event_pp"), T("adv", "today")),)
        if s.name == "NP": return ((T("det", s.role + "_det"), T("adj_person", s.role + "_adj"), T("noun_person", s.role + "_noun")),)
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
    out = []
    def walk(n: Node):
        if n.symbol.name == "T": out.append(Slot(len(out), n.symbol))
        else:
            for child in n.children: walk(child)
    walk(tree); return tuple(out)


def replay_ledger(ledger: list[dict[str, object]]) -> dict[str, object]:
    residual = ""; cancellations = 0
    for event in ledger:
        char = str(event["char"])
        if residual:
            if char != residual[0]: return {"ok": False, "events_replayed": len(ledger), "cancellations": cancellations, "residual": residual}
            residual = residual[1:]; cancellations += 1
        else: residual = char
    return {"ok": not residual, "events_replayed": len(ledger), "cancellations": cancellations, "residual": residual}


def trace(center: str, pivot: int, left_outer: str, right_outer: str) -> dict[str, object]:
    left = [(center, center[:pivot][::-1]), (left_outer, left_outer[::-1])]; right = [(center, center[pivot:]), (right_outer, right_outer)]
    cur = {"left": None, "right": None}; pos = {"left": [0, 0], "right": [0, 0]}; residual = ""; owner = ""; preferred = "left"; ledger = []
    while True:
        for side in ("left", "right"):
            if cur[side] is not None and pos[side][1] >= len(cur[side][1]): cur[side] = None
        if all(cur[s] is None and pos[s][0] >= len(left if s == "left" else right) for s in ("left", "right")): break
        side = ("right" if owner == "left" else "left") if residual else preferred
        if cur[side] is None:
            leaves = left if side == "left" else right
            if pos[side][0] >= len(leaves): return {"completed": False, "ledger": ledger, "replay": replay_ledger(ledger), "first_incompatible": {"side": side, "reason": "frontier exhausted"}}
            i = pos[side][0]; cur[side] = leaves[i]; pos[side] = [i + 1, 0]
        word, stream = cur[side]; char = stream[pos[side][1]]; pos[side][1] += 1; event = {"side": side, "word": word, "char": char, "residual_before": residual}
        if residual and char != residual[0]:
            event["action"] = "contradiction"; ledger.append(event); return {"completed": False, "ledger": ledger, "replay": replay_ledger(ledger), "first_incompatible": {"side": side, "word": word, "char": char, "expected": residual[0], "residual": residual}}
        if residual: residual = residual[1:]; event["action"] = "cancel"
        else: residual = char; owner = side; event["action"] = "open"
        event["residual_after"] = residual; ledger.append(event)
        if not residual: preferred = "left"
    return {"completed": not residual, "ledger": ledger, "replay": replay_ledger(ledger), "cancellations": sum(e["action"] == "cancel" for e in ledger)}


def centre_inventory(stats: Counter) -> list[dict[str, object]]:
    rows = []
    for item in BY_KIND["noun_center"]:
        if item.word not in PIVOTS: continue
        pivots = []
        for pivot in range(1, len(item.word)):
            left, right = item.word[:pivot][::-1], item.word[pivot:]; common = 0
            while common < min(len(left), len(right)) and left[common] == right[common]: common += 1
            pivots.append({"pivot": pivot, "split": item.word[:pivot] + "|" + item.word[pivot:], "internal_prefix_matches": common})
        stats["centres_enumerated"] += 1; rows.append({"word": item.word, "meaning": item.meaning, "pivots": pivots})
    return rows


def boundary_inventory(stats: Counter) -> list[dict[str, object]]:
    rows = []
    for center in BY_KIND["noun_center"]:
        if center.word not in PIVOTS: continue
        pivot = PIVOTS[center.word]; frame = FRAME[center.meaning]
        for adj, relation in itertools.product(BY_KIND["adj_center"], BY_KIND["prep"]):
            if adj.word not in frame["adjectives"] or relation.word not in frame["relations"]: continue
            stats["semantic_boundary_triples_enumerated"] += 1
            tr = trace(center.word, pivot[0], adj.word, relation.word)
            rows.append({"center": center.word, "pivot": f"{center.word[:pivot[0]]}|{center.word[pivot[0]:]}", "adjacent_adjective": adj.word, "relation": relation.word, "trace": tr, "cancellations": tr["replay"]["cancellations"], "events": len(tr["ledger"])})
    return sorted(rows, key=lambda row: (-int(row["cancellations"]), int(row["events"]), row["center"], row["adjacent_adjective"]))


def semantic_choice(slot: Slot, item: Lexeme, assigned: dict[str, Lexeme]) -> bool:
    role = slot.symbol.role
    if role == "center_word": return item.meaning in FRAME
    if role == "center_adj":
        center = assigned.get("center_word"); return center is not None and item.word in FRAME[center.meaning]["adjectives"]
    if role == "intro_verb":
        center = assigned.get("center_word"); return center is not None and item.word == FRAME[center.meaning]["verb"]
    if role == "relation":
        center = assigned.get("center_word"); return center is not None and item.word in FRAME[center.meaning]["relations"]
    if role == "s2_subject_noun" and "s1_subject_noun" in assigned: return item.word == assigned["s1_subject_noun"].word
    if role == "record_verb": return item.object_type == "record"
    return True


def determiner_ok(words: list[str], leaf_slots: tuple[Slot, ...]) -> bool:
    return all(not (slot.symbol.kind == "det" and words[i] in {"a", "an"} and ((words[i] == "an") != (words[i + 1][:1] in "aeiou"))) for i, slot in enumerate(leaf_slots[:-1]))


def parse_tree(grammar: Grammar, text: str) -> bool:
    if text != text.strip() or re.sub(r"[A-Za-z ,.?!'-]", "", text): return False
    tokens = tuple(WORD.findall(text.lower())); leaf_slots = slots(grammar.expand(Symbol("S")))
    if len(tokens) != len(leaf_slots): return False
    assigned = {}
    for i, slot in enumerate(leaf_slots):
        item = next((x for x in BY_KIND.get(slot.symbol.kind, ()) if x.word == tokens[i]), None)
        if item is None: return False
        assigned[slot.symbol.role] = item
    center = assigned["center_word"]
    return (determiner_ok(list(tokens), leaf_slots) and center.meaning in FRAME and
            assigned["center_adj"].word in FRAME[center.meaning]["adjectives"] and
            assigned["intro_verb"].word == FRAME[center.meaning]["verb"] and
            assigned["relation"].word in FRAME[center.meaning]["relations"] and
            assigned["s1_subject_noun"].meaning == "person" and
            assigned["s2_subject_noun"].word == assigned["s1_subject_noun"].word and
            assigned["record_verb"].object_type == "record")


def render(words: list[str]) -> str:
    first, second = " ".join(words[:11]), " ".join(words[11:]); return first[:1].upper() + first[1:] + ". " + second[:1].upper() + second[1:] + "."


def audit(grammar: Grammar, text: str, kind: str, provenance: tuple[str, ...]) -> dict[str, object]:
    tape = normalize_letters(text); gate = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS); parsed = parse_tree(grammar, text); codes = [k for k, v in gate.items() if not v]
    if not parsed: codes.append("independent_complete_reparse_failed")
    return {"record_kind": kind, "rendered": text, "provenance": provenance, "independent_exact_audit": {"exact": bool(tape) and tape == tape[::-1], "letters": len(tape), "normalized_sha256": sha256(tape.encode()).hexdigest()}, "independent_parse": parsed, "central_admission": gate, "mechanically_admitted": not codes, "rejection_codes": codes, "reader_status": "unreviewed; programmatic checks do not certify readability"}


def search(grammar: Grammar, leaf_slots: tuple[Slot, ...], *, state_limit: int, closure_limit: int, stats: Counter) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    center = next(slot.index for slot in leaf_slots if slot.symbol.role == "center_word"); pivots = PIVOTS
    words = [""] * len(leaf_slots); exact, admitted = [], []
    def visit(left_next, right_next, left_stream, right_stream, assigned, residual, owner, last_side, ledger):
        if stats["state_count"] >= state_limit or len(exact) >= closure_limit: return
        stats["search_states"] += 1
        if left_stream is not None and left_stream[2] >= len(left_stream[1]): left_stream = None
        if right_stream is not None and right_stream[2] >= len(right_stream[1]): right_stream = None
        if left_next < 0 and right_next >= len(leaf_slots) and left_stream is None and right_stream is None:
            stats["complete_tree_states"] += 1; text = render(words)
            if not determiner_ok(words, leaf_slots): stats["agreement_rejections"] += 1; return
            row = audit(grammar, text, "complete_fresh_internal_center_tree", tuple(words))
            if row["independent_exact_audit"]["exact"]:
                exact.append(row); stats["exact_closures"] += 1
                if row["mechanically_admitted"]:
                    admitted.append(row); stats["admitted_closures"] += 1
            return
        def available(side):
            stream = left_stream if side == "left" else right_stream; nxt = left_next if side == "left" else right_next
            return stream is not None or (nxt >= 0 if side == "left" else nxt < len(leaf_slots))
        if residual:
            sides = ("right",) if owner == "left" else ("left",)
            if not available(sides[0]): stats["residual_frontier_exhaustions"] += 1; return
        else: sides = tuple(side for side in (last_side, "right" if last_side == "left" else "left") if available(side))
        for side in sides:
            stream = left_stream if side == "left" else right_stream; next_index = left_next if side == "left" else right_next
            if stream is None:
                slot = leaf_slots[next_index]
                for item in BY_KIND[slot.symbol.kind]:
                    if not semantic_choice(slot, item, assigned): continue
                    assigned[slot.symbol.role] = item; words[next_index] = item.word; stats["lexical_assignments_considered"] += 1
                    chars = normalize_letters(item.word); new_stream = (next_index, chars[::-1] if side == "left" else chars, 0)
                    visit(left_next - 1 if side == "left" else left_next, right_next + 1 if side == "right" else right_next, new_stream if side == "left" else left_stream, new_stream if side == "right" else right_stream, assigned, residual, owner, side, ledger)
                    words[next_index] = ""; assigned.pop(slot.symbol.role, None)
                continue
            index, chars, pos = stream; char = chars[pos]; event = {"side": side, "slot": index, "word": words[index], "char": char, "residual_before": residual}; stats["state_count"] += 1; stats["character_emissions"] += 1
            if residual and char != residual[0]:
                event.update(action="contradiction", expected=residual[0]); stats["residual_contradictions"] += 1; stats.setdefault("first_incompatible", {"side": side, "word": words[index], "char": char, "expected": residual[0]})
                if len(ledger) + 1 > stats.get("deepest_ledger_length", 0):
                    stats["deepest_ledger_length"] = len(ledger) + 1
                    stats["deepest_live_ledger"] = ledger[:]
                    stats["deepest_next_literal_rejection"] = event
                    stats["deepest_independent_replay"] = replay_ledger(ledger)
                continue
            if residual: residual2, owner2 = residual[1:], ""; event["action"] = "cancel"; stats["residual_cancellations"] += 1
            else: residual2, owner2 = char, side; event["action"] = "open"
            event["residual_after"] = residual2; new_stream = (index, chars, pos + 1)
            visit(left_next, right_next, new_stream if side == "left" else left_stream, new_stream if side == "right" else right_stream, assigned, residual2, owner2, side, ledger + [event])
    for center_item in BY_KIND[leaf_slots[center].symbol.kind]:
        if center_item.word not in pivots: continue
        words[center] = center_item.word; assigned = {leaf_slots[center].symbol.role: center_item}; stats["lexical_assignments_considered"] += 1; chars = normalize_letters(center_item.word); pivot = pivots[center_item.word][0]
        visit(center - 1, center + 1, (center, chars[:pivot][::-1], 0), (center, chars[pivot:], 0), assigned, "", "", "left", [])
        words[center] = ""
    return exact, admitted


def run(*, state_limit: int = 100_000, closure_limit: int = 100) -> dict[str, object]:
    grammar = Grammar(); leaf_slots = slots(grammar.expand(Symbol("S"))); stats = Counter(state_count=0, search_states=0, lexical_assignments_considered=0, character_emissions=0, residual_cancellations=0, residual_contradictions=0, complete_tree_states=0, exact_closures=0, admitted_closures=0)
    inventory = centre_inventory(stats); boundaries = boundary_inventory(stats); best = boundaries[0] if boundaries else None; exact, admitted = search(grammar, leaf_slots, state_limit=state_limit, closure_limit=closure_limit, stats=stats)
    controls = [audit(grammar, "The careful researcher described a public attack during the annual trial. The careful researcher documented the annual report after the public study today.", "shared_referent_two_sentence_control", tuple(WORD.findall("the careful researcher described a public attack during the annual trial the careful researcher documented the annual report after the public study today"))), audit(grammar, "The patient analyst measured a visible effect during the annual trial. The patient analyst reviewed the public record after the western study today.", "shared_referent_two_sentence_control", tuple(WORD.findall("the patient analyst measured a visible effect during the annual trial the patient analyst reviewed the public record after the western study today")))]
    deepest = {"ledger_before_rejection": stats.get("deepest_live_ledger", []), "next_literal_rejection": stats.get("deepest_next_literal_rejection"), "independent_replay": stats.get("deepest_independent_replay"), "emissions_including_rejection": stats.get("deepest_ledger_length", 0)}
    return {"status": "noncareer_internal_center_typed_tree_search", "config": {"state_limit": state_limit, "closure_limit": closure_limit, "one_connected_tree": True, "grammar_owns_every_leaf": True, "word_internal_center_inventory": True, "prepared_multiword_center": False, "joint_boundary_enumeration_before_search": True, "career_pivot_family_retired": True, "one_character_emission_states": True, "replayed_ledger": True, "independent_complete_reparse": True, "reject_every_self_palindromic_contiguous_multiword_span": True, "corpus_or_catalogue_generation": False}, "grammar_leaf_count": len(leaf_slots), "centre_inventory": inventory, "boundary_inventory": boundaries, "best_boundary_trace": best, "deepest_full_scheduler_replay": deepest, "stats": dict(stats), "exact_closures": exact, "admitted_closures": admitted, "complete_grammar_controls": controls, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(), "material": "task-authored ordinary noncareer center inventory and shared-referent discourse grammar; no catalogue text"}, "reader_facing_next_operator": "Use the persisted deepest scheduler mismatch from the noncareer frame to choose the next grammar topology; do not revive the retired career pivot family or hand-edit a preflight trace.", "scope": "The inventory and bounded full-tree ledger are construction diagnostics; no readability claim is made without blinded readers."}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); p.add_argument("--state-limit", type=int, default=100_000); p.add_argument("--closure-limit", type=int, default=100); args = p.parse_args()
    if args.out.exists(): p.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit, closure_limit=args.closure_limit); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["state_count"], "exact": len(result["exact_closures"]), "admitted": len(result["admitted_closures"])}, indent=2))


if __name__ == "__main__": main()
