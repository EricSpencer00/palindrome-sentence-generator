"""Natural locative-topology repair after the ``hall:l``/``extra:r`` seam.

The previous subject modifier ``extra`` was an ordinary adjective but its
final edge exposed ``r`` after the matched ``ha``.  This branch changes the
collective-person NP topology to an authored locative attribution:
``the villa help`` (household help associated with a villa).  The locative
noun ``villa`` ends in ``la``, so its reverse exposes ``a`` then ``l`` and
crosses the former conflict.  The complete probe is replayed through the
shared emitter until its next incompatibility; it is diagnostic, never a
palindrome candidate.
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
SOURCE = ROOT / "experiments/inverted_question_natural_location_bridge_replay_20260913.py"
spec = importlib.util.spec_from_file_location("natural_location_for_villa_help_20260913", SOURCE)
NATURAL = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = NATURAL
spec.loader.exec_module(NATURAL)
BASE, QUESTION, EDGE = NATURAL.BASE, NATURAL.QUESTION, NATURAL.EDGE
FIXED_LABELS = NATURAL.FIXED_LABELS
MIN_LETTERS, MAX_LETTERS = NATURAL.MIN_LETTERS, NATURAL.MAX_LETTERS

VILLA = EDGE.EdgeLexeme("villa", "noun", "location_modifier", units=("vil", "la"), rank=3)
MORPHOLOGY = NATURAL.MORPHOLOGY + (VILLA,)
LEXEME_BY_FORM = {x.form: x for x in MORPHOLOGY}


class VillaHelpGrammar(NATURAL.NaturalLocationGrammar):
    """Natural location grammar with a typed ``villa``-attributed help NP."""

    def productions(self, lhs: BASE.Symbol) -> tuple[BASE.Production, ...]:
        f = dict(lhs.features)
        if lhs.name == "NP" and f.get("type") == "person_group":
            d = int(f["depth"])
            villa_help = BASE.Production(f"NP:relative_agent:person_group:{d}:villa-help", lhs, (
                BASE.sym("DET_SLOT", number="plur", form_role=f["role"]),
                BASE.sym("LEX_SLOT", category="noun", type="location_modifier", role="subject_modifier", number="sing"),
                BASE.sym("LEX_SLOT", category="noun", type="person_group", role=f["role"], number="plur"),
            ))
            return super().productions(lhs) + (villa_help,)
        if lhs.name == "LEX_SLOT" and f.get("type") == "location_modifier" and f.get("role") == "subject_modifier":
            return (BASE.Production("LEX_SLOT:villa", lhs, (
                BASE.sym("T", label="noun_subject_modifier", form="villa"),
            )),)
        return super().productions(lhs)


def choose(grammar: VillaHelpGrammar, state: BASE.State, index: int, ident: str) -> BASE.State:
    options = [x for x in BASE.expand(grammar, state, index) if x.trace[-1][1] == ident]
    if len(options) != 1:
        raise AssertionError(f"nonunique replay production {ident}")
    return options[0]


def build_probe_tree(grammar: VillaHelpGrammar) -> BASE.State:
    state = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ())
    while True:
        nodes = BASE.node_map(state)
        unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        if not unresolved:
            break
        index = unresolved[0]; symbol = nodes[state.frontier[index]].symbol; f = dict(symbol.features)
        if symbol.name == "Q":
            ident = "Q:was-it-location-i-saw"
        elif symbol.name == "NP":
            if f.get("type") == "location":
                ident = "NP:complement:location:1:compound"
            else:
                ident = f"NP:relative_agent:person_group:{f['depth']}:villa-help"
        elif symbol.name == "DET_SLOT":
            ident = "DET_SLOT:the" if f.get("number") == "plur" else "DET_SLOT:a"
        elif symbol.name == "LEX_SLOT":
            if f.get("role") == "compound_modifier":
                form = "temple"
            elif f.get("role") == "subject_modifier":
                form = "villa"
            elif f.get("type") == "location":
                form = "hall"
            elif f.get("type") == "person_group":
                form = "help"
            else:
                form = "met"
            ident = f"LEX_SLOT:{form}"
        elif symbol.name == "REL":
            ident = "REL:location:0:locative-group"
        elif symbol.name == "V":
            ident = "V:locative_met"
        else:
            raise AssertionError(symbol.name)
        state = choose(grammar, state, index, ident)
    if any(not BASE.node_map(state)[ref].terminal for ref in state.frontier):
        raise AssertionError("probe tree was not fully expanded")
    return state


def replay_bridge(grammar: VillaHelpGrammar) -> dict[str, object]:
    state = build_probe_tree(VillaHelpGrammar(1)); text = BASE.render(state)
    emitted, events = state, []
    # Consume the fixed ``wasi`` envelope pairwise, then continue through the
    # shared tree until the first failed opposite-edge emission.
    sides = (1, 1, 1, 1, -1, -1, -1, -1)
    for side in sides:
        nodes, leaves = BASE.node_map(emitted), BASE.leaf_map(emitted)
        ref = next(r for r in (emitted.frontier if side == 1 else reversed(emitted.frontier)) if nodes[r].terminal)
        leaf = leaves[ref]; char = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]
        emitted = BASE.emit(emitted, side)
        if emitted is None:
            raise AssertionError("fixed envelope replay failed")
    step = len(sides)
    failed_attempt = None
    while step < 80:
        side = 1 if step % 2 == 0 else -1
        nodes, leaves = BASE.node_map(emitted), BASE.leaf_map(emitted)
        active = [r for r in (emitted.frontier if side == 1 else reversed(emitted.frontier)) if nodes[r].terminal]
        if not active:
            break
        leaf = leaves[active[0]]; char = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]
        next_state = BASE.emit(emitted, side); step += 1
        if next_state is None:
            failed_attempt = {"step": step, "side": side, "slot": leaf.label, "word": leaf.word,
                              "character": char, "emitter_rejected": True}
            break
        emitted = next_state
        if step >= 9:
            events.append({"step": step, "side": side, "slot": leaf.label, "word": leaf.word,
                           "character": char, "residual_after": emitted.residual, "owner_after": emitted.owner})
    pairs = [(events[i]["character"], events[i + 1]["character"])
             for i in range(0, len(events) - 1, 2)]
    return {"rendered": text, "bridge_pairs": pairs,
            "bridge_stream": "".join(x[0] for x in pairs),
            "terminal_reverse_stream": "".join(x[1] for x in pairs),
            "emitter_events": events, "failed_attempt": failed_attempt,
            "pairs_replayed": len(pairs),
            "villa_boundary_crossed": any(x["word"] == "villa" for x in events),
            "first_post_repair_mismatch": failed_attempt,
            "independent_parse": QUESTION.parse_tree(grammar, text) is not None,
            "exact_audit": QUESTION.exact_audit(text), "diagnostic_only": True}


def content_words(state: BASE.State) -> tuple[str, ...]:
    return tuple(x.word for x in BASE.ordered_leaves(state) if x.label not in FIXED_LABELS)


def lexical_unit_accepts(state: BASE.State, side: int, added: list[BASE.Leaf]) -> bool:
    if not state.residual or state.owner == side:
        return True
    expected = state.residual[0]
    for leaf in added:
        if leaf.label in FIXED_LABELS:
            continue
        lexeme = LEXEME_BY_FORM.get(leaf.word)
        if lexeme is None:
            return False
        unit = EDGE.active_unit(lexeme, leaf, side)
        if unit[0 if side == 1 else -1] != expected:
            return False
    return True


def expand_edge(grammar: VillaHelpGrammar, state: BASE.State, index: int, side: int) -> tuple[BASE.State, ...]:
    used = set(content_words(state)); previous = BASE.leaf_map(state); output = []
    for candidate in BASE.expand(grammar, state, index):
        added = [leaf for leaf in candidate.leaves if leaf.identifier not in previous]
        if not lexical_unit_accepts(state, side, added):
            continue
        if any(leaf.word in used or leaf.word == leaf.word[::-1]
               for leaf in added if leaf.label not in FIXED_LABELS):
            continue
        output.append(candidate)
    return tuple(output)


def solver(grammar: VillaHelpGrammar, *, max_states: int = 100000) -> dict[str, object]:
    root = BASE.Node(0, grammar.start()); initial = BASE.State((0,), (root,), (), "", 0, 0, ())
    queue, seen, exact_rows, admitted, rejected = deque([initial]), set(), {}, {}, {}
    stats = Counter(states=0, expansions=0, emissions=0, residual_contradictions=0,
                    complete_trees=0, exact_closures=0, independent_reparse_rejections=0)
    while queue and stats["states"] < max_states:
        state = queue.pop(); stats["states"] += 1
        key = (state.frontier, state.nodes, state.leaves, state.residual, state.owner, state.length)
        if key in seen:
            continue
        seen.add(key)
        if BASE.complete(state):
            stats["complete_trees"] += 1
            if MIN_LETTERS <= state.length <= MAX_LETTERS and state.residual == state.residual[::-1]:
                text = BASE.render(state)
                row = NATURAL.audit(grammar, text, "complete_villa_help_bridge_closure", state.trace)
                if row["independent_exact_audit"]["exact"] and row["independent_parse"]:
                    stats["exact_closures"] += 1; exact_rows.setdefault(text, row)
                    if row["mechanically_admitted"]:
                        admitted.setdefault(text, row)
                    elif len(rejected) < 100:
                        rejected.setdefault(text, row); stats["independent_reparse_rejections"] += 1
            continue
        nodes = BASE.node_map(state)
        unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        index = None
        if state.residual:
            side = -state.owner; edge = 0 if side == 1 else len(state.frontier) - 1
            if not nodes[state.frontier[edge]].terminal:
                index = edge
        elif unresolved:
            if not nodes[state.frontier[0]].terminal:
                index, side = 0, 1
            elif not nodes[state.frontier[-1]].terminal:
                index, side = len(state.frontier) - 1, -1
            else:
                index, side = unresolved[0], 1
        if index is not None:
            for next_state in expand_edge(grammar, state, index, side):
                queue.append(next_state); stats["expansions"] += 1
        left_ready = bool(nodes[state.frontier[0]].terminal); right_ready = bool(nodes[state.frontier[-1]].terminal)
        sides = (-state.owner,) if state.residual else (1, -1)
        for side in sides:
            if (side == 1 and not left_ready) or (side == -1 and not right_ready):
                continue
            next_state = BASE.emit(state, side)
            if next_state is not None:
                queue.append(next_state); stats["emissions"] += 1
            else:
                stats["residual_contradictions"] += 1
    return {"stats": dict(stats), "states_exhausted": not queue and stats["states"] < max_states,
            "exact_closures": list(exact_rows.values()),
            "mechanically_admitted_closures": list(admitted.values()),
            "independent_reparse_rejections": list(rejected.values())}


def run(*, max_states: int = 100000) -> dict[str, object]:
    grammar = VillaHelpGrammar(6); result = solver(grammar, max_states=max_states); replay = replay_bridge(grammar)
    control_state = NATURAL.SECOND.SECOND.LOCATIVE.explicit_control(grammar)
    control_text = BASE.render(control_state)
    control = NATURAL.audit(grammar, control_text, "complete_villa_help_recursive_control", control_state.trace)
    content = [x.word for x in BASE.ordered_leaves(control_state) if x.label not in FIXED_LABELS]
    control.update({"diagnostic_only": True, "grammar_tree_fully_expanded": True, "content_word_forms": content,
                    "content_word_forms_unique": len(content) == len(set(content)),
                    "provenance": {"construction": "explicit six-level natural location control", "shared_tree": True},
                    "reader_status": "grammar control only; not a palindrome candidate or readability evidence"})
    return {"status": "inverted_question_natural_location_villa_help_replay",
            "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_depth": grammar.max_depth,
                       "max_states": max_states, "single_shared_derivation_tree": True,
                       "character_residual_during_derivation": True, "programmatic_emitter_replay": True,
                       "collective_subject_semantic_license": "villa help = household help associated with a villa",
                       "locative_topology_repair": "location-attributed collective-person NP",
                       "closure_requires_all_leaves_consumed": True, "independent_complete_reparse": True,
                       "corpus_generation": False},
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "grammar_sha256": grammar.digest(),
                           "material": "authored temple-hall/villa-help locative topology; no catalogue or known palindrome material"},
            "replayed_boundary_preflight": replay, "complete_recursive_control": control,
            "edge_unit_inventory": {"entries": [asdict(x) for x in MORPHOLOGY]}, **result,
            "reader_facing_next_operator": "Use the captured post-repair mismatch to alter the licensed relative boundary, with another real emitter replay.",
            "scope": "Replay and control are diagnostic only; only fresh >=100-letter exact closures enter reader evaluation."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(max_states=args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["states"],
                      "exact_closures": len(result["exact_closures"]),
                      "admitted": len(result["mechanically_admitted_closures"])}, indent=2))


if __name__ == "__main__":
    main()
