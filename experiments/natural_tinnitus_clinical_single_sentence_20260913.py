"""Frame-first, no-repeated-content construction for a clinical event.

The complete natural frame is fixed before search: chronic tinnitus causes
persistent sleep loss while clinicians investigate underlying hearing damage in
a quiet clinic today.  The word-internal ``tin|nitus`` boundary is then derived
from that frame; no shared-subject sentence stack or prepared palindrome is
used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

MIN_LETTERS, MAX_LETTERS = 100, 260


class Lexeme:
    def __init__(self, word: str, kind: str, meaning: str): self.word, self.kind, self.meaning = word, kind, meaning


class Symbol:
    def __init__(self, name: str, role: str = "", kind: str = ""): self.name, self.role, self.kind = name, role, kind
    def __hash__(self): return hash((self.name, self.role, self.kind))
    def __eq__(self, other): return isinstance(other, Symbol) and (self.name, self.role, self.kind) == (other.name, other.role, other.kind)


class Node:
    def __init__(self, symbol: Symbol, children: tuple["Node", ...] = ()): self.symbol, self.children = symbol, children


class Slot:
    def __init__(self, index: int, symbol: Symbol): self.index, self.symbol = index, symbol


def T(kind: str, role: str) -> Symbol: return Symbol("T", role, kind)


LEXICON = (
    Lexeme("chronic", "adj_center", "tinnitus"), Lexeme("tinnitus", "noun_center", "tinnitus"), Lexeme("causes", "verb_predicate", "causation"),
    Lexeme("persistent", "adj_target", "loss"), Lexeme("sleep", "noun_target", "loss"), Lexeme("loss", "noun_complement", "loss"),
    Lexeme("while", "conj", "contrast"), Lexeme("careful", "adj_person", "clinician"), Lexeme("skilled", "adj_person", "clinician"), Lexeme("clinicians", "noun_person", "clinician"), Lexeme("investigate", "verb_action", "investigation"),
    Lexeme("underlying", "adj_artifact", "damage"), Lexeme("hearing", "adj_artifact2", "damage"), Lexeme("damage", "noun_artifact", "damage"), Lexeme("in", "prep", "location"), Lexeme("a", "det", "indefinite"), Lexeme("quiet", "adj_event", "clinic"), Lexeme("clinic", "noun_event", "clinic"), Lexeme("today", "adv", "temporal"),
)
BY_KIND: dict[str, tuple[Lexeme, ...]] = {}
for item in LEXICON: BY_KIND[item.kind] = BY_KIND.get(item.kind, ()) + (item,)


class Grammar:
    def productions(self, s: Symbol) -> tuple[tuple[Symbol, ...], ...]:
        if s.name == "S": return ((T("adj_center", "center_adj"), T("noun_center", "center_word"), T("verb_predicate", "predicate"), T("adj_target", "target_adj"), T("noun_target", "target_noun"), T("noun_complement", "complement"), T("conj", "subordinator"), T("adj_person", "agent_adj"), T("noun_person", "agent_noun"), T("verb_action", "action"), T("adj_artifact", "object_adj"), T("adj_artifact2", "object_adj2"), T("noun_artifact", "object_noun"), T("prep", "location_prep"), T("det", "location_det"), T("adj_event", "location_adj"), T("noun_event", "location_noun"), T("adv", "today")),)
        return ()
    def expand(self, s: Symbol) -> Node:
        rhs = self.productions(s); return Node(s, tuple(self.expand(child) for child in rhs[0])) if rhs else Node(s)
    def digest(self) -> str:
        rows, seen, queue = [], set(), [Symbol("S")]
        while queue:
            s = queue.pop(0)
            if s in seen: continue
            seen.add(s); rhs = self.productions(s); rows.append((s.name, s.role, s.kind, [[(x.name, x.role, x.kind) for x in row] for row in rhs])); queue.extend(x for row in rhs for x in row if x.name != "T")
        return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def slots(tree: Node) -> tuple[Slot, ...]:
    out = []
    def walk(node: Node):
        if node.symbol.name == "T": out.append(Slot(len(out), node.symbol))
        else:
            for child in node.children: walk(child)
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


def derived_pivot(word: str) -> tuple[int, int]:
    scores = {}
    for pivot in range(1, len(word)):
        left, right = word[:pivot][::-1], word[pivot:]; common = 0
        while common < min(len(left), len(right)) and left[common] == right[common]: common += 1
        scores[pivot] = common
    pivot = max(scores, key=lambda p: (scores[p], -p)); return pivot, scores[pivot]


FRAME = {"center": "tinnitus", "pivot": "tin|nitus", "center_adjective": "chronic", "predicate": "causes", "target": ("persistent", "sleep", "loss"), "subordinator": "while", "agent_adjectives": {"careful", "skilled"}, "agent_noun": "clinicians", "action": "investigate", "object": ("underlying", "hearing", "damage"), "location": ("in", "a", "quiet", "clinic"), "today": "today"}


def semantic_choice(slot: Slot, item: Lexeme) -> bool:
    allowed = {"center_adj": {FRAME["center_adjective"]}, "center_word": {FRAME["center"]}, "predicate": {FRAME["predicate"]}, "target_adj": {FRAME["target"][0]}, "target_noun": {FRAME["target"][1]}, "complement": {FRAME["target"][2]}, "subordinator": {FRAME["subordinator"]}, "agent_adj": FRAME["agent_adjectives"], "agent_noun": {FRAME["agent_noun"]}, "action": {FRAME["action"]}, "object_adj": {FRAME["object"][0]}, "object_adj2": {FRAME["object"][1]}, "object_noun": {FRAME["object"][2]}, "location_prep": {FRAME["location"][0]}, "location_det": {FRAME["location"][1]}, "location_adj": {FRAME["location"][2]}, "location_noun": {FRAME["location"][3]}, "today": {FRAME["today"]}}
    return item.word in allowed.get(slot.symbol.role, set())


def render(words: list[str]) -> str: return " ".join(words).capitalize() + "."


def parse_tree(grammar: Grammar, text: str) -> bool:
    if text != text.strip() or re.sub(r"[A-Za-z ,.?!'-]", "", text): return False
    tokens = tokenize(text); leaf_slots = slots(grammar.expand(Symbol("S")))
    return len(tokens) == len(leaf_slots) and all((item := next((x for x in BY_KIND.get(slot.symbol.kind, ()) if x.word == tokens[i]), None)) is not None and semantic_choice(slot, item) for i, slot in enumerate(leaf_slots))


def audit(grammar: Grammar, text: str, kind: str, provenance: tuple[str, ...]) -> dict[str, object]:
    tape = normalize_letters(text); gate = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS); parsed = parse_tree(grammar, text); codes = [key for key, value in gate.items() if not value]
    if not parsed: codes.append("independent_complete_reparse_failed")
    return {"record_kind": kind, "rendered": text, "provenance": provenance, "independent_exact_audit": {"exact": bool(tape) and tape == tape[::-1], "letters": len(tape), "normalized_sha256": sha256(tape.encode()).hexdigest()}, "independent_parse": parsed, "central_admission": gate, "mechanically_admitted": not codes, "rejection_codes": codes, "reader_status": "unreviewed; programmatic checks do not certify readability"}


def search(grammar: Grammar, leaf_slots: tuple[Slot, ...], *, state_limit: int, closure_limit: int, stats: Counter):
    center = next(slot.index for slot in leaf_slots if slot.symbol.role == "center_word"); center_word = FRAME["center"]; pivot, _ = derived_pivot(center_word); words = [""] * len(leaf_slots); exact, admitted = [], []
    def visit(left_next, right_next, left_stream, right_stream, assigned, residual, owner, last_side, ledger):
        if stats["state_count"] >= state_limit or len(exact) >= closure_limit: return
        stats["search_states"] += 1
        if left_stream is not None and left_stream[2] >= len(left_stream[1]): left_stream = None
        if right_stream is not None and right_stream[2] >= len(right_stream[1]): right_stream = None
        if left_next < 0 and right_next >= len(leaf_slots) and left_stream is None and right_stream is None:
            stats["complete_tree_states"] += 1; row = audit(grammar, render(words), "complete_natural_no_repeat_tree", tuple(words))
            if row["independent_exact_audit"]["exact"]:
                exact.append(row); stats["exact_closures"] += 1
                if row["mechanically_admitted"]: admitted.append(row); stats["admitted_closures"] += 1
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
                for item in BY_KIND.get(slot.symbol.kind, ()):
                    if not semantic_choice(slot, item): continue
                    assigned[slot.symbol.role] = item; words[next_index] = item.word; stats["lexical_assignments_considered"] += 1; chars = normalize_letters(item.word); new_stream = (next_index, chars[::-1] if side == "left" else chars, 0)
                    visit(left_next - 1 if side == "left" else left_next, right_next + 1 if side == "right" else right_next, new_stream if side == "left" else left_stream, new_stream if side == "right" else right_stream, assigned, residual, owner, side, ledger)
                    words[next_index] = ""; assigned.pop(slot.symbol.role, None)
                continue
            index, chars, position = stream; char = chars[position]; event = {"side": side, "slot": index, "word": words[index], "char": char, "residual_before": residual}; stats["state_count"] += 1; stats["character_emissions"] += 1
            if residual and char != residual[0]:
                event.update(action="contradiction", expected=residual[0]); stats["residual_contradictions"] += 1
                if len(ledger) + 1 > stats.get("deepest_ledger_length", 0): stats["deepest_ledger_length"] = len(ledger) + 1; stats["deepest_live_ledger"] = ledger[:]; stats["deepest_next_literal_rejection"] = event; stats["deepest_independent_replay"] = replay_ledger(ledger)
                continue
            if residual: residual2, owner2 = residual[1:], ""; event["action"] = "cancel"; stats["residual_cancellations"] += 1
            else: residual2, owner2 = char, side; event["action"] = "open"
            event["residual_after"] = residual2; new_stream = (index, chars, position + 1); visit(left_next, right_next, new_stream if side == "left" else left_stream, new_stream if side == "right" else right_stream, assigned, residual2, owner2, side, ledger + [event])
    center_item = next(item for item in BY_KIND["noun_center"] if item.word == center_word); words[center] = center_word; assigned = {leaf_slots[center].symbol.role: center_item}; stats["lexical_assignments_considered"] += 1; visit(center - 1, center + 1, (center, normalize_letters(center_word[:pivot])[::-1], 0), (center, normalize_letters(center_word[pivot:]), 0), assigned, "", "", "left", []); words[center] = ""; return exact, admitted


def run(*, state_limit: int = 100_000, closure_limit: int = 100) -> dict[str, object]:
    grammar = Grammar(); leaf_slots = slots(grammar.expand(Symbol("S"))); pivot, matches = derived_pivot(FRAME["center"]); stats = Counter(state_count=0, search_states=0, lexical_assignments_considered=0, character_emissions=0, residual_cancellations=0, residual_contradictions=0, complete_tree_states=0, exact_closures=0, admitted_closures=0)
    frames = [{"rendered": f"Chronic tinnitus causes persistent sleep loss while {agent} clinicians investigate underlying hearing damage in a quiet clinic today.", "center": FRAME["center"], "pivot": FRAME["pivot"], "predicate": "causes", "semantic_status": "complete ordinary clinical event fixed before center search"} for agent in ("careful", "skilled")]
    exact, admitted = search(grammar, leaf_slots, state_limit=state_limit, closure_limit=closure_limit, stats=stats); controls = [audit(grammar, frame["rendered"], "complete_natural_no_repeat_control", tuple(tokenize(frame["rendered"]))) for frame in frames]; deepest = {"ledger_before_rejection": stats.get("deepest_live_ledger", []), "next_literal_rejection": stats.get("deepest_next_literal_rejection"), "independent_replay": stats.get("deepest_independent_replay"), "emissions_including_rejection": stats.get("deepest_ledger_length", 0)}
    return {"status": "natural_tinnitus_clinical_single_sentence_no_repeat_typed_tree_search", "config": {"state_limit": state_limit, "closure_limit": closure_limit, "one_connected_tree": True, "grammar_owns_every_leaf": True, "complete_event_frame_before_center_search": True, "frame_first_inventory": True, "no_repeated_content_topology": True, "post_center_predicate_semantically_selected": True, "word_internal_center_inventory": True, "pivot_derived_from_center_boundary": True, "prepared_multiword_center": False, "one_character_emission_states": True, "replayed_ledger": True, "independent_complete_reparse": True, "reject_every_self_palindromic_contiguous_multiword_span": True, "corpus_or_catalogue_generation": False}, "grammar_leaf_count": len(leaf_slots), "derived_center": {"word": FRAME["center"], "pivot": pivot, "split": f"{FRAME['center'][:pivot]}|{FRAME['center'][pivot:]}", "initial_matches": matches}, "sentence_frame_inventory": frames, "deepest_full_scheduler_replay": deepest, "stats": dict(stats), "exact_closures": exact, "admitted_closures": admitted, "complete_grammar_controls": controls, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(), "material": "task-authored complete natural clinical event frames; no catalogue text"}, "reader_facing_next_operator": "Derive the next repair from this no-repeat topology's actual replayed frontier; do not patch the clinical predicate or revive retired center families.", "scope": "This bounded construction run records exact diagnostics; programmatic controls do not certify human readability."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--state-limit", type=int, default=100_000); parser.add_argument("--closure-limit", type=int, default=100); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit, closure_limit=args.closure_limit); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["state_count"], "exact": len(result["exact_closures"]), "admitted": len(result["admitted_closures"])}, indent=2))


if __name__ == "__main__": main()
