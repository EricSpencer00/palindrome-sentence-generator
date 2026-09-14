"""Natural location-boundary repair with an executable mismatch capture.

The prior replay stopped at ``tatempleh`` because the relative boundary
``that`` (``t``) met the subject determiner's reverse ``e``.  This branch
changes the licensed location phrase to ``a temple hall entrance``.  The
executable replay records the actual next failure at the hall boundary
(``a`` versus ``e``) without inventing a continuation.  The entrance head
is retained in the grammar and independently parsed, but is not claimed to
have been reached by this character replay.
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
spec = importlib.util.spec_from_file_location("natural_location_for_entrance_replay_20260913", SOURCE)
NATURAL = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = NATURAL
spec.loader.exec_module(NATURAL)
BASE, QUESTION = NATURAL.BASE, NATURAL.QUESTION
EDGE = NATURAL.EDGE
FIXED_LABELS = NATURAL.FIXED_LABELS
MIN_LETTERS, MAX_LETTERS = NATURAL.MIN_LETTERS, NATURAL.MAX_LETTERS
ENTRANCE = EDGE.EdgeLexeme("entrance", "noun", "location", units=("ent", "rance"), rank=3)
MORPHOLOGY = NATURAL.MORPHOLOGY + (ENTRANCE,)
LEXEME_BY_FORM = dict(NATURAL.LEXEME_BY_FORM, entrance=ENTRANCE)


class EntranceGrammar(NATURAL.NaturalLocationGrammar):
    """Typed three-slot location phrase with an entrance head."""

    def productions(self, lhs: BASE.Symbol) -> tuple[BASE.Production, ...]:
        f = dict(lhs.features)
        if lhs.name == "NP" and f.get("type") == "location" and int(f["depth"]) > 0:
            base = super().productions(lhs); d = int(f["depth"])
            return base + (BASE.Production(f"NP:{f['role']}:location:{d}:compound-entrance", lhs, (
                BASE.sym("DET_SLOT", number="sing", form_role=f["role"]),
                BASE.sym("LEX_SLOT", category="noun", type="location_modifier", role="compound_modifier", number="sing"),
                BASE.sym("LEX_SLOT", category="noun", type="location", role="hall_modifier", number="sing"),
                BASE.sym("LEX_SLOT", category="noun", type="location", role=f["role"], number="sing"),
                BASE.sym("REL", head_type="location", depth=str(d - 1)))),)
        if lhs.name == "LEX_SLOT" and lhs.feature("role") == "hall_modifier":
            return (BASE.Production("LEX_SLOT:hall", lhs, (BASE.sym("T", label="noun_hall_modifier", form="hall"),)),)
        if lhs.name == "LEX_SLOT" and lhs.feature("type") == "location" and lhs.feature("role") == "complement":
            return super().productions(lhs) + (BASE.Production("LEX_SLOT:entrance", lhs, (BASE.sym("T", label="noun_complement", form="entrance"),)),)
        return super().productions(lhs)


def choose(grammar: EntranceGrammar, state: BASE.State, index: int, ident: str) -> BASE.State:
    options = [x for x in BASE.expand(grammar, state, index) if x.trace[-1][1] == ident]
    if len(options) != 1: raise AssertionError(f"nonunique production {ident}")
    return options[0]


def build_probe_tree(grammar: EntranceGrammar) -> BASE.State:
    state = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ())
    while True:
        nodes = BASE.node_map(state); unresolved = [i for i, r in enumerate(state.frontier) if not nodes[r].terminal]
        if not unresolved: break
        i = unresolved[0]; s = nodes[state.frontier[i]].symbol; f = dict(s.features)
        if s.name == "Q": ident = "Q:was-it-location-i-saw"
        elif s.name == "NP": ident = "NP:complement:location:1:compound-entrance" if f.get("type") == "location" else f"NP:{f['role']}:person_group:0:plain"
        elif s.name == "DET_SLOT": ident = "DET_SLOT:the" if f.get("number") == "plur" else "DET_SLOT:a"
        elif s.name == "LEX_SLOT":
            form = "temple" if f.get("role") == "compound_modifier" else "hall" if f.get("role") == "hall_modifier" else "entrance" if f.get("type") == "location" else "help" if f.get("type") == "person_group" else "met"
            ident = f"LEX_SLOT:{form}"
        elif s.name == "REL": ident = "REL:location:0:locative-group"
        elif s.name == "V": ident = "V:locative_met"
        else: raise AssertionError(s.name)
        state = choose(grammar, state, i, ident)
    return state


def replay_bridge(grammar: EntranceGrammar) -> dict[str, object]:
    state = build_probe_tree(EntranceGrammar(1)); text = BASE.render(state)
    sides = (1, 1, 1, 1, -1, -1, -1, -1) + tuple(x for _ in range(9) for x in (1, -1))
    emitted, events = state, []
    for step, side in enumerate(sides, 1):
        nodes, leaves = BASE.node_map(emitted), BASE.leaf_map(emitted); ref = next(r for r in (emitted.frontier if side == 1 else reversed(emitted.frontier)) if nodes[r].terminal); leaf = leaves[ref]
        char = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]; emitted = BASE.emit(emitted, side)
        if emitted is None: raise AssertionError(f"unexpected preflight failure at {step}")
        if step >= 9: events.append({"step": step, "side": side, "slot": leaf.label, "word": leaf.word, "character": char, "residual_after": emitted.residual})
    left_nodes, left_leaves = BASE.node_map(emitted), BASE.leaf_map(emitted); left_ref = next(r for r in emitted.frontier if left_nodes[r].terminal); left_leaf = left_leaves[left_ref]; left_char = left_leaf.word[left_leaf.left]
    after_left = BASE.emit(emitted, 1)
    if after_left is None: raise AssertionError("left mismatch should create residual")
    right_nodes, right_leaves = BASE.node_map(after_left), BASE.leaf_map(after_left); right_ref = next(r for r in reversed(after_left.frontier) if right_nodes[r].terminal); right_leaf = right_leaves[right_ref]; right_char = right_leaf.word[-1 - right_leaf.right]
    failed = BASE.emit(after_left, -1)
    return {"rendered": text, "bridge_pairs": [(events[i]["character"], events[i + 1]["character"]) for i in range(0, len(events), 2)], "bridge_stream": "".join(events[i]["character"] for i in range(0, len(events), 2)), "terminal_reverse_stream": "".join(events[i + 1]["character"] for i in range(0, len(events), 2)), "emitter_events": events, "nine_pairs_replayed": len(events) == 18, "compound_boundary_crossed": events[-2]["word"] == "hall" and events[-2]["character"] == "h", "first_post_bridge_attempt": {"left_slot": left_leaf.label, "left_word": left_leaf.word, "left_character": left_char, "right_slot": right_leaf.label, "right_word": right_leaf.word, "right_character": right_char, "emitter_rejected": failed is None}, "independent_parse": QUESTION.parse_tree(grammar, text) is not None, "exact_audit": QUESTION.exact_audit(text), "diagnostic_only": True}


def content_words(state: BASE.State) -> tuple[str, ...]: return tuple(x.word for x in BASE.ordered_leaves(state) if x.label not in FIXED_LABELS)


def expand_edge(grammar: EntranceGrammar, state: BASE.State, index: int, side: int) -> tuple[BASE.State, ...]:
    used = set(content_words(state)); previous = BASE.leaf_map(state); out = []
    for candidate in BASE.expand(grammar, state, index):
        added = [x for x in candidate.leaves if x.identifier not in previous]
        if state.residual and state.owner != side:
            expected = state.residual[0]
            for leaf in added:
                if leaf.label in FIXED_LABELS: continue
                lexeme = LEXEME_BY_FORM.get(leaf.word)
                if lexeme and EDGE.active_unit(lexeme, leaf, side)[0 if side == 1 else -1] != expected: break
            else:
                if not any(x.word in used or x.word == x.word[::-1] for x in added if x.label not in FIXED_LABELS): out.append(candidate)
        elif not any(x.word in used or x.word == x.word[::-1] for x in added if x.label not in FIXED_LABELS): out.append(candidate)
    return tuple(out)


def solver(grammar: EntranceGrammar, *, max_states: int = 100000) -> dict[str, object]:
    # Reuse the validated single-tree traversal with this branch's lexical
    # filter, so newly exposed entrance slots obey the same residual rule.
    root = BASE.Node(0, grammar.start()); initial = BASE.State((0,), (root,), (), "", 0, 0, ())
    queue, seen, exact_rows, admitted, rejected = deque([initial]), set(), {}, {}, {}; stats = Counter(states=0, expansions=0, emissions=0, residual_contradictions=0, complete_trees=0, exact_closures=0, independent_reparse_rejections=0)
    while queue and stats["states"] < max_states:
        state = queue.pop(); stats["states"] += 1; key = (state.frontier, state.nodes, state.leaves, state.residual, state.owner, state.length)
        if key in seen: continue
        seen.add(key)
        if BASE.complete(state):
            stats["complete_trees"] += 1
            if MIN_LETTERS <= state.length <= MAX_LETTERS and state.residual == state.residual[::-1]:
                text = BASE.render(state); row = audit(grammar, text, "complete_entrance_bridge_closure", state.trace)
                if row["independent_exact_audit"]["exact"] and row["independent_parse"]:
                    stats["exact_closures"] += 1; exact_rows.setdefault(text, row)
                    if row["mechanically_admitted"]: admitted.setdefault(text, row)
                    elif len(rejected) < 100: rejected.setdefault(text, row); stats["independent_reparse_rejections"] += 1
            continue
        nodes = BASE.node_map(state); unresolved = [i for i, r in enumerate(state.frontier) if not nodes[r].terminal]; index = None
        if state.residual:
            side = -state.owner; edge = 0 if side == 1 else len(state.frontier) - 1
            if not nodes[state.frontier[edge]].terminal: index = edge
        elif unresolved:
            if not nodes[state.frontier[0]].terminal: index, side = 0, 1
            elif not nodes[state.frontier[-1]].terminal: index, side = len(state.frontier) - 1, -1
            else: index, side = unresolved[0], 1
        if index is not None:
            for nxt in expand_edge(grammar, state, index, side): queue.append(nxt); stats["expansions"] += 1
        left, right = bool(nodes[state.frontier[0]].terminal), bool(nodes[state.frontier[-1]].terminal); dirs = (-state.owner,) if state.residual else (1, -1)
        for side in dirs:
            if (side == 1 and not left) or (side == -1 and not right): continue
            nxt = BASE.emit(state, side)
            if nxt is not None: queue.append(nxt); stats["emissions"] += 1
            else: stats["residual_contradictions"] += 1
    return {"stats": dict(stats), "states_exhausted": not queue and stats["states"] < max_states, "exact_closures": list(exact_rows.values()), "mechanically_admitted_closures": list(admitted.values()), "independent_reparse_rejections": list(rejected.values())}


def run(*, max_states: int = 100000) -> dict[str, object]:
    grammar = EntranceGrammar(6); result = solver(grammar, max_states=max_states); replay = replay_bridge(grammar)
    control_state = NATURAL.SECOND.SECOND.LOCATIVE.explicit_control(grammar); control_text = BASE.render(control_state)
    control = NATURAL.audit(grammar, control_text, "complete_entrance_recursive_control", control_state.trace)
    content = [x.word for x in BASE.ordered_leaves(control_state) if x.label not in FIXED_LABELS]
    control.update({"diagnostic_only": True, "grammar_tree_fully_expanded": True, "content_word_forms": content,
                    "content_word_forms_unique": len(content) == len(set(content)),
                    "provenance": {"construction": "explicit six-level natural location control", "shared_tree": True},
                    "reader_status": "grammar control only; not a palindrome candidate or readability evidence"})
    return {"status": "inverted_question_natural_location_entrance_replay", "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_depth": grammar.max_depth, "max_states": max_states, "single_shared_derivation_tree": True, "character_residual_during_derivation": True, "programmatic_emitter_replay": True, "nine_pair_bridge": True, "natural_compound_semantic_license": "temple modifier + hall intermediate + entrance location head", "closure_requires_all_leaves_consumed": True, "independent_complete_reparse": True, "corpus_generation": False}, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(), "material": "authored temple-hall-entrance/help locative bridge; no catalogue or known palindrome material"}, "replayed_boundary_preflight": replay, "complete_recursive_control": control, "edge_unit_inventory": {"entries": [asdict(x) for x in MORPHOLOGY]}, **result, "reader_facing_next_operator": "Use the captured a-versus-e mismatch to design the next licensed relation boundary; do not claim a continuation without replay.", "scope": "The replay is diagnostic only; only fresh >=100-letter exact closures enter reader evaluation."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(max_states=args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact_closures": len(result["exact_closures"]), "admitted": len(result["mechanically_admitted_closures"])}, indent=2))


if __name__ == "__main__": main()
