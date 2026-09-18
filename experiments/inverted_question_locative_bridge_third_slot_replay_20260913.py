"""Third-slot extension with executable emitter replay.

The six-pair ``tatemp`` replay is extended by an authored compound location:
``a tempo hall that a bishop met at``.  ``tempo`` is a typed object modifier
and ``hall`` is the typed location head.  The actual emitter reaches eight
matched pairs (``tatempoh``), with the eighth pair entering the third lexical
slot.  No known palindrome or catalogue phrase is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/inverted_question_locative_bridge_second_slot_replay_20260913.py"
spec = importlib.util.spec_from_file_location("second_replay_for_third_slot_20260913", SOURCE)
SECOND = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = SECOND
spec.loader.exec_module(SECOND)
BASE, QUESTION = SECOND.BASE, SECOND.QUESTION

TEMPO = SECOND.SECOND.LOCATIVE.EDGE.EdgeLexeme("tempo", "noun", "object", units=("tem", "po"), rank=7)
HALL = SECOND.SECOND.LOCATIVE.EDGE.EdgeLexeme("hall", "noun", "location", units=("hall",), rank=2)
MORPHOLOGY = SECOND.SECOND.MORPHOLOGY + (TEMPO, HALL)
LEXEME_BY_FORM = dict(SECOND.SECOND.LEXEME_BY_FORM, tempo=TEMPO, hall=HALL)
FIXED_LABELS = SECOND.SECOND.FIXED_LABELS


class ThirdSlotGrammar(SECOND.SECOND.SecondSlotGrammar):
    """Second-slot grammar with a typed compound location production."""

    def productions(self, lhs: BASE.Symbol) -> tuple[BASE.Production, ...]:
        f = dict(lhs.features)
        if lhs.name == "NP" and f.get("type") == "location":
            base = super().productions(lhs)
            d = int(f["depth"])
            if d > 0:
                return base + (BASE.Production(f"NP:{f['role']}:location:{d}:compound", lhs, (
                    BASE.sym("DET_SLOT", number=f["number"], form_role=f["role"]),
                    BASE.sym("LEX_SLOT", category="noun", type="object", role="compound_modifier", number=f["number"]),
                    BASE.sym("LEX_SLOT", category="noun", type="location", role=f["role"], number=f["number"]),
                    BASE.sym("REL", head_type="location", depth=str(d - 1)))),)
            return base
        if lhs.name == "LEX_SLOT" and lhs.feature("type") == "object" and lhs.feature("role") == "compound_modifier":
            return (BASE.Production("LEX_SLOT:tempo", lhs, (
                BASE.sym("T", label="noun_compound_modifier", form="tempo"),)),)
        if lhs.name == "LEX_SLOT" and lhs.feature("type") == "location":
            base = super().productions(lhs)
            return base + (BASE.Production("LEX_SLOT:hall", lhs, (
                BASE.sym("T", label=f"noun_{lhs.feature('role')}", form="hall"),)),)
        return super().productions(lhs)


def choose(grammar: ThirdSlotGrammar, state: BASE.State, index: int, identifier: str) -> BASE.State:
    options = [x for x in BASE.expand(grammar, state, index) if x.trace[-1][1] == identifier]
    if len(options) != 1: raise AssertionError(f"nonunique third-slot production {identifier}")
    return options[0]


def build_probe_tree(grammar: ThirdSlotGrammar) -> BASE.State:
    state = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ())
    while True:
        nodes = BASE.node_map(state); unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        if not unresolved: break
        index = unresolved[0]; symbol = nodes[state.frontier[index]].symbol; f = dict(symbol.features)
        if symbol.name == "Q": ident = "Q:was-it-location-i-saw"
        elif symbol.name == "NP": ident = f"NP:{f['role']}:location:1:compound" if f.get("type") == "location" else f"NP:{f['role']}:{f['type']}:0:plain"
        elif symbol.name == "DET_SLOT": ident = "DET_SLOT:a"
        elif symbol.name == "LEX_SLOT":
            form = ("tempo" if f.get("role") == "compound_modifier" else
                    "hall" if f.get("type") == "location" else
                    "met" if f.get("category") == "verb" else "bishop")
            ident = f"LEX_SLOT:{form}"
        elif symbol.name == "REL": ident = "REL:location:0:locative"
        elif symbol.name == "V": ident = "V:locative_met"
        else: raise AssertionError(symbol.name)
        state = choose(grammar, state, index, ident)
    if any(not BASE.node_map(state)[ref].terminal for ref in state.frontier): raise AssertionError("probe not fully expanded")
    return state


def replay_bridge(grammar: ThirdSlotGrammar) -> dict[str, object]:
    # The probe itself has exactly one compound relative; it is independently
    # reparsed against the larger depth used by the search.
    state = build_probe_tree(ThirdSlotGrammar(1)); text = BASE.render(state)
    sides = (1, 1, 1, 1, -1, -1, -1, -1) + tuple(x for _ in range(8) for x in (1, -1))
    emitted, events = state, []
    for step, side in enumerate(sides, start=1):
        nodes, leaves = BASE.node_map(emitted), BASE.leaf_map(emitted)
        ref = next(ref for ref in (emitted.frontier if side == 1 else reversed(emitted.frontier)) if nodes[ref].terminal)
        leaf = leaves[ref]; char = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]
        emitted = BASE.emit(emitted, side)
        if emitted is None: raise AssertionError(f"emitter rejected step {step}")
        if step >= 9: events.append({"step": step, "side": side, "slot": leaf.label, "word": leaf.word, "character": char, "residual_after": emitted.residual})
    paired = [(events[i], events[i + 1]) for i in range(0, len(events), 2)]
    pairs = [(a["character"], b["character"]) for a, b in paired]
    return {"rendered": text, "bridge_pairs": pairs, "bridge_stream": "".join(x[0] for x in pairs),
            "terminal_reverse_stream": "".join(x[1] for x in pairs), "emitter_events": events,
            "eight_events_replayed": len(events) == 16,
            "third_slot_event": events[-2]["slot"] == "noun_complement" and events[-2]["word"] == "hall",
            "independent_parse": QUESTION.parse_tree(grammar, text) is not None,
            "exact_audit": QUESTION.exact_audit(text), "diagnostic_only": True}


def lexical_unit_accepts(state: BASE.State, side: int, added: list[BASE.Leaf]) -> bool:
    if not state.residual or state.owner == side: return True
    expected = state.residual[0]
    for leaf in added:
        if leaf.label in FIXED_LABELS: continue
        lexeme = LEXEME_BY_FORM.get(leaf.word)
        if lexeme and SECOND.SECOND.LOCATIVE.EDGE.active_unit(lexeme, leaf, side)[0 if side == 1 else -1] != expected: return False
    return True


def content_words(state: BASE.State) -> tuple[str, ...]:
    return tuple(x.word for x in BASE.ordered_leaves(state) if x.label not in FIXED_LABELS)


def expand_edge(grammar: ThirdSlotGrammar, state: BASE.State, index: int, side: int) -> tuple[BASE.State, ...]:
    used = set(content_words(state)); previous = BASE.leaf_map(state); out = []
    for candidate in BASE.expand(grammar, state, index):
        added = [x for x in candidate.leaves if x.identifier not in previous]
        if not lexical_unit_accepts(state, side, added): continue
        if any(x.word in used or x.word == x.word[::-1] for x in added if x.label not in FIXED_LABELS): continue
        out.append(candidate)
    return tuple(out)


def solver(grammar: ThirdSlotGrammar, *, max_states: int = 100000) -> dict[str, object]:
    root = BASE.Node(0, grammar.start()); initial = BASE.State((0,), (root,), (), "", 0, 0, ())
    queue, seen, exact_rows, admitted, rejected = deque([initial]), set(), {}, {}, {}
    stats = Counter(states=0, expansions=0, emissions=0, residual_contradictions=0, complete_trees=0, exact_closures=0, independent_reparse_rejections=0)
    while queue and stats["states"] < max_states:
        state = queue.pop(); stats["states"] += 1; key = (state.frontier, state.nodes, state.leaves, state.residual, state.owner, state.length)
        if key in seen: continue
        seen.add(key)
        if BASE.complete(state):
            stats["complete_trees"] += 1
            if MIN_LETTERS <= state.length <= MAX_LETTERS and state.residual == state.residual[::-1]:
                text = BASE.render(state); row = SECOND.SECOND.audit(grammar, text, "complete_third_slot_closure", state.trace)
                if row["independent_exact_audit"]["exact"] and row["independent_parse"]:
                    stats["exact_closures"] += 1; exact_rows.setdefault(text, row)
                    if row["mechanically_admitted"]: admitted.setdefault(text, row)
                    elif len(rejected) < 100: rejected.setdefault(text, row); stats["independent_reparse_rejections"] += 1
            continue
        nodes = BASE.node_map(state); unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]; index = None
        if state.residual:
            side = -state.owner; edge = 0 if side == 1 else len(state.frontier) - 1
            if not nodes[state.frontier[edge]].terminal: index = edge
        elif unresolved:
            if not nodes[state.frontier[0]].terminal: index, side = 0, 1
            elif not nodes[state.frontier[-1]].terminal: index, side = len(state.frontier) - 1, -1
            else: index, side = unresolved[0], 1
        if index is not None:
            for nxt in expand_edge(grammar, state, index, side): queue.append(nxt); stats["expansions"] += 1
        left, right = bool(nodes[state.frontier[0]].terminal), bool(nodes[state.frontier[-1]].terminal)
        sides2 = (-state.owner,) if state.residual else (1, -1)
        for side in sides2:
            if (side == 1 and not left) or (side == -1 and not right): continue
            nxt = BASE.emit(state, side)
            if nxt is not None: queue.append(nxt); stats["emissions"] += 1
            else: stats["residual_contradictions"] += 1
    return {"stats": dict(stats), "states_exhausted": not queue and stats["states"] < max_states, "exact_closures": list(exact_rows.values()), "mechanically_admitted_closures": list(admitted.values()), "independent_reparse_rejections": list(rejected.values())}


MIN_LETTERS, MAX_LETTERS = SECOND.SECOND.MIN_LETTERS, SECOND.SECOND.MAX_LETTERS


def run(*, max_states: int = 100000) -> dict[str, object]:
    grammar = ThirdSlotGrammar(6); result = solver(grammar, max_states=max_states); replay = replay_bridge(grammar)
    control_state = SECOND.SECOND.LOCATIVE.explicit_control(grammar); control_text = BASE.render(control_state)
    control = SECOND.SECOND.audit(grammar, control_text, "complete_third_slot_recursive_control", control_state.trace)
    content = [x.word for x in BASE.ordered_leaves(control_state) if x.label not in FIXED_LABELS]
    control.update({"diagnostic_only": True, "grammar_tree_fully_expanded": True, "content_word_forms": content,
                    "content_word_forms_unique": len(content) == len(set(content)), "provenance": {"construction": "explicit six-level locative tree", "lexical_source": "hand-authored edge-unit inventory only", "shared_tree": True},
                    "reader_status": "grammar control only; not a palindrome candidate or readability evidence"})
    return {"status": "inverted_question_locative_third_slot_replay", "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_depth": grammar.max_depth, "max_states": max_states, "single_shared_derivation_tree": True, "character_residual_during_derivation": True, "programmatic_emitter_replay": True, "third_typed_slot_bridge": True, "closure_requires_all_leaves_consumed": True, "independent_complete_reparse": True, "corpus_generation": False},
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(), "material": "authored tempo/hall/bishop locative bridge; no catalogue or known palindrome material"},
            "replayed_boundary_preflight": replay, "complete_recursive_control": control, "edge_unit_inventory": {"entries": [asdict(x) for x in MORPHOLOGY]}, **result,
            "reader_facing_next_operator": "Only after this eight-pair replay, add another independently authored typed lexical span.", "scope": "The replay is bridge diagnostic evidence only; only fresh >=100-letter exact closures enter reader evaluation."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(max_states=args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact_closures": len(result["exact_closures"]), "admitted": len(result["mechanically_admitted_closures"])}, indent=2))


if __name__ == "__main__": main()
