"""Replay-validated second-slot locative bridge.

This branch turns the prior declarative ``tatemp`` trace into an executable
replay of the one connected probe tree.  It expands the exact grammar path,
then calls the shared character emitter in the same left/right order used by
the search.  The six bridge pairs are retained only as a preflight diagnostic;
the >=100-letter run still reports only independently reparsed full closures.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/inverted_question_locative_bridge_second_slot_20260913.py"
spec = importlib.util.spec_from_file_location("second_slot_for_replay_20260913", SOURCE)
SECOND = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = SECOND
spec.loader.exec_module(SECOND)
BASE, QUESTION = SECOND.BASE, SECOND.QUESTION


def choose(grammar: SECOND.SecondSlotGrammar, state: BASE.State, index: int, identifier: str) -> BASE.State:
    options = [x for x in BASE.expand(grammar, state, index) if x.trace[-1][1] == identifier]
    if len(options) != 1: raise AssertionError(f"nonunique replay production {identifier}")
    return options[0]


def build_probe_tree(grammar: SECOND.SecondSlotGrammar) -> BASE.State:
    """Build exactly ``a temple that a bishop met at`` as one tree."""
    state = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ())
    while True:
        nodes = BASE.node_map(state)
        unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        if not unresolved: break
        index = unresolved[0]; symbol = nodes[state.frontier[index]].symbol; f = dict(symbol.features)
        if symbol.name == "Q": ident = "Q:was-it-location-i-saw"
        elif symbol.name == "NP":
            d = int(f["depth"]); shape = "plain-relative" if d > 0 else "plain"
            ident = f"NP:{f['role']}:{f['type']}:{d}:{shape}"
        elif symbol.name == "DET_SLOT": ident = "DET_SLOT:a"
        elif symbol.name == "LEX_SLOT":
            form = "temple" if f.get("type") == "location" else ("bishop" if f.get("type") == "person" else "met")
            ident = f"LEX_SLOT:{form}"
        elif symbol.name == "REL": ident = f"REL:location:{f['depth']}:locative"
        elif symbol.name == "V": ident = "V:locative_met"
        else: raise AssertionError(symbol.name)
        state = choose(grammar, state, index, ident)
    if any(not BASE.node_map(state)[ref].terminal for ref in state.frontier):
        raise AssertionError("probe tree was not fully expanded")
    return state


def replay_bridge(grammar: SECOND.SecondSlotGrammar) -> dict[str, object]:
    # The probe intentionally has one relative (the bridge itself); the
    # production path is then reparsed against the larger search grammar.
    state = build_probe_tree(SECOND.SecondSlotGrammar(1))
    text = BASE.render(state)
    # First consume ``wasi`` from the left and its reverse from the right.
    sides = (1, 1, 1, 1, -1, -1, -1, -1,
             1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1)
    events, emitted = [], state
    for step, side in enumerate(sides, start=1):
        nodes, leaves = BASE.node_map(emitted), BASE.leaf_map(emitted)
        active_ref = next(ref for ref in (emitted.frontier if side == 1 else reversed(emitted.frontier))
                          if nodes[ref].terminal)
        leaf = leaves[active_ref]
        char = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]
        before = emitted
        emitted = BASE.emit(emitted, side)
        if emitted is None: raise AssertionError(f"emitter rejected replay step {step}")
        if step >= 9:
            events.append({"step": step, "side": side, "slot": leaf.label, "word": leaf.word, "character": char,
                           "residual_after": emitted.residual, "owner_after": emitted.owner})
    paired = [(events[i], events[i + 1]) for i in range(0, len(events), 2)]
    bridge_pairs = [(left["character"], right["character"]) for left, right in paired]
    return {"rendered": text, "normalized": "".join(c for c in text if "a" <= c <= "z"),
            "replayed_sides": list(sides), "emitter_events": events, "bridge_pairs": bridge_pairs,
            "bridge_stream": "".join(x[0] for x in bridge_pairs),
            "terminal_reverse_stream": "".join(x[1] for x in bridge_pairs),
            "six_events_replayed": len(events) == 12,
            "second_slot_event": events[-1]["slot"] == "noun_relative_agent" and events[-1]["word"] == "bishop",
            "independent_parse": QUESTION.parse_tree(grammar, text) is not None,
            "exact_audit": QUESTION.exact_audit(text), "diagnostic_only": True}


def run(*, max_states: int = 100000) -> dict[str, object]:
    grammar = SECOND.SecondSlotGrammar(6); result = SECOND.solver(grammar, max_states=max_states)
    replay = replay_bridge(grammar)
    control_state = SECOND.LOCATIVE.explicit_control(grammar); control_text = BASE.render(control_state)
    control = SECOND.audit(grammar, control_text, "complete_replay_recursive_control", control_state.trace)
    content = [x.word for x in BASE.ordered_leaves(control_state) if x.label not in SECOND.FIXED_LABELS]
    control.update({"diagnostic_only": True, "grammar_tree_fully_expanded": True, "content_word_forms": content,
                    "content_word_forms_unique": len(content) == len(set(content)),
                    "provenance": {"construction": "explicit six-level locative relative tree",
                                   "lexical_source": "hand-authored edge-unit inventory only", "shared_tree": True},
                    "reader_status": "grammar control only; not a palindrome candidate or readability evidence"})
    return {"status": "inverted_question_locative_second_slot_replay", "config": {
        "min_letters": SECOND.MIN_LETTERS, "max_letters": SECOND.MAX_LETTERS, "max_depth": grammar.max_depth,
        "max_states": max_states, "single_shared_derivation_tree": True, "character_residual_during_derivation": True,
        "programmatic_emitter_replay": True, "second_typed_slot_bridge": True,
        "closure_requires_all_leaves_consumed": True, "independent_complete_reparse": True, "corpus_generation": False},
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(),
                       "material": "authored temple/met-at/bishop edge-unit bridge; no catalogue or known palindrome material"},
        "replayed_boundary_preflight": replay, "complete_recursive_control": control,
        "edge_unit_inventory": {"entries": [asdict(x) for x in SECOND.MORPHOLOGY]}, **result,
        "reader_facing_next_operator": "Only after this replay audit, extend the bridge with a third authored typed slot and replay every event.",
        "scope": "The replay is a bridge diagnostic, not candidate or readability evidence; only fresh >=100-letter exact closures enter reader evaluation."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000)
    args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(max_states=args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact_closures": len(result["exact_closures"]), "admitted": len(result["mechanically_admitted_closures"])}, indent=2))


if __name__ == "__main__": main()
