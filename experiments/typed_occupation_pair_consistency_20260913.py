"""Joint consistency for two exposed lexical roles plus occupation morphology.

This repairs the observed teacher/met conflict. Both exposed role slots are
made structurally visible before a neutral lexical choice; incompatible word
overlaps are pruned jointly. No interior slot is lexicalized. Normal clipped
and agentive occupation forms add a compatible person realization (temp/met)
without appending phrases or self-palindromic units. Every retained pair is
still emitted one character at a time through the original exact zipper.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/relational_interior_bridge_20260913.py"
SPEC = importlib.util.spec_from_file_location("occupation_pair_bridge", SOURCE)
BRIDGE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = BRIDGE
SPEC.loader.exec_module(BRIDGE)
PARENT, BASE = BRIDGE.PARENT, BRIDGE.BASE
OCCUPATIONS = {
    "temp": {"role": "person", "formation": "lexical clipping", "gloss": "temporary worker"},
    "deliverer": {"role": "person", "formation": "deliver + er", "gloss": "person who delivers"},
    "reporter": {"role": "person", "formation": "report + er", "gloss": "person who reports"},
    "trainer": {"role": "person", "formation": "train + er", "gloss": "person who trains"},
    "waiter": {"role": "person", "formation": "wait + er", "gloss": "person who serves diners"},
    "sailor": {"role": "person", "formation": "sail + or", "gloss": "person who sails"},
}


class Grammar(BRIDGE.BridgeGrammar):
    def productions(self, lhs):
        original = super().productions(lhs)
        if lhs.name == "W" and lhs.feature("category") == "person":
            return original + tuple(BASE.Production("occupation:"+form, lhs,
                (BASE.sym("T", form=form, label="person"),)) for form in OCCUPATIONS)
        return original


def overlap_compatible(left, right):
    reverse = right[::-1]
    count = min(len(left), len(reverse))
    return left[:count] == reverse[:count]


def successors(grammar, state, stats):
    if state.residual or len(state.frontier) < 2:
        return PARENT.successors(grammar, state, stats)
    nodes = BASE.node_map(state)
    left, right = nodes[state.frontier[0]], nodes[state.frontier[-1]]
    if left.symbol.name != "W":
        return PARENT.successors(grammar, state, stats)
    if not right.terminal and right.symbol.name != "W":
        # Only expose structure on the other boundary. No right word is
        # selected before its own W slot reaches that boundary.
        expanded = BASE.expand(grammar, state, len(state.frontier)-1)
        assert all(len(candidate.leaves) == len(state.leaves) for candidate in expanded)
        stats["opposite_structural_expansions"] += len(expanded)
        return expanded
    if right.symbol.name != "W":
        return PARENT.successors(grammar, state, stats)
    output = []
    for left_state in PARENT.successors(grammar, state, stats):
        left_word = BASE.leaf_map(left_state)[left_state.frontier[0]].word
        prior = BASE.leaf_map(left_state)
        used = {leaf.word for leaf in left_state.leaves if leaf.word not in PARENT.REPEATABLE_FUNCTION_WORDS}
        for paired in BASE.expand(grammar, left_state, len(left_state.frontier)-1):
            added = [leaf for leaf in paired.leaves if leaf.identifier not in prior]
            assert len(added) == 1
            right_word = added[0].word
            if right_word not in PARENT.REPEATABLE_FUNCTION_WORDS and (right_word in used or right_word == right_word[::-1]):
                continue
            if overlap_compatible(left_word, right_word):
                output.append(paired)
                stats["exposed_word_pairs_retained"] += 1
            else:
                stats["exposed_word_pairs_rejected"] += 1
    return tuple(output)


CONTROL = ("reward a temp with a rare medal and carry the detailed portrait of the patient teacher "
           "beside the old drawing to the careful artist who met a drawer")


def emitted_control_trace(grammar, text):
    """Replay an independently parsed diagnostic through real exposed edges.

    The control's parse chooses productions solely for this verification;
    neither it nor this replay supplies any search initial state or word pair.
    """
    tree = BASE.parse_tree(grammar, text)
    assert tree is not None
    mapping = {0: tree}
    state = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ())
    emissions = []
    while not BASE.complete(state):
        side, edge = PARENT.active_edge(state)
        nodes = BASE.node_map(state)
        node = nodes[state.frontier[edge]]
        if not node.terminal:
            parse_node = mapping[node.identifier]
            options = BASE.expand(grammar, state, edge)
            state = next(candidate for candidate in options if candidate.trace[-1][1] == parse_node.production)
            parent = BASE.node_map(state)[node.identifier]
            mapping.update(zip(parent.children, parse_node.children))
            continue
        leaf = BASE.leaf_map(state)[node.identifier]
        char = leaf.word[leaf.left] if side == 1 else leaf.word[-1-leaf.right]
        following = BASE.emit(state, side)
        if following is None:
            return {"diagnostic_only": True, "emissions": emissions, "emitted_letters": state.length,
                    "matched_pairs": state.length//2, "failed_side": side,
                    "failed_character": char, "live_debt": state.residual,
                    "complete_palindrome": False}
        emissions.append({"side": side, "word": leaf.word, "character": char,
                          "debt_before": state.residual, "debt_after": following.residual})
        state = following
    return {"diagnostic_only": True, "emissions": emissions, "emitted_letters": state.length,
            "matched_pairs": state.length//2, "complete_palindrome": True}


def solve(grammar, max_states):
    stack = [BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ())]
    stats = Counter(states=0, complete_trees=0, deepest_emitted_letters=0)
    exact, admitted = {}, {}
    while stack and stats["states"] < max_states:
        state = stack.pop()
        stats["states"] += 1
        stats["deepest_emitted_letters"] = max(stats["deepest_emitted_letters"], state.length)
        if state.length > PARENT.MAX_LETTERS:
            continue
        if BASE.complete(state):
            stats["complete_trees"] += 1
            if state.residual == state.residual[::-1]:
                text = BASE.render(state)
                row = PARENT.audit(grammar, text, "complete_shared_tree_closure", state.trace)
                assert row["independent_exact_audit"]["exact"]
                exact.setdefault(text, row)
                if row["mechanically_admitted"]:
                    admitted.setdefault(text, row)
            continue
        stack.extend(reversed(successors(grammar, state, stats)))
    return {"stats": dict(stats), "states_exhausted": not stack,
            "exact_closures": list(exact.values()), "mechanically_admitted_closures": list(admitted.values())}


def run(max_states=100000):
    grammar = Grammar()
    control = PARENT.audit(grammar, CONTROL, "intact_prose_grammar_control")
    control["diagnostic_only"] = True
    trace = emitted_control_trace(grammar, CONTROL)
    assert trace["emitted_letters"] > 19
    assert control["independent_parse"] and control["independent_exact_audit"]["letters"] > 100
    return {"method": "typed_occupation_morphology_with_exposed_pair_consistency",
            "construction_change": "replace the incompatible teacher/met occupation realization with a compatible temp/met overlap and jointly constrain two exposed lexical roles before character emission",
            "config": {"max_states": max_states, "single_shared_tree": True,
                       "words_only_at_exposed_leaves": True, "character_equality_during_emission": True},
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "bridge_sha256": sha256(SOURCE.read_bytes()).hexdigest(),
                           "grammar_sha256": grammar.digest(), "occupation_forms": OCCUPATIONS,
                           "catalogue_generation_material": False, "control_used_as_search_seed": False},
            "complete_grammar_control": control, "cross_19_emitted_control_trace": trace,
            **solve(grammar, max_states),
            "next_reader_facing_test": "Any long admitted closure requires randomized blinded human reading with intact and shuffled controls, including paraphrase and grammaticality judgments.",
            "next_construction_if_no_candidate": "Extend paired role consistency across the next syntactic boundary with alternate complete relative-clause forms, rather than add unmatched occupation words."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-states", default=100000, type=int)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite an existing artifact")
    result = run(args.max_states)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"out": str(args.out), "stats": result["stats"], "exact": len(result["exact_closures"]),
                      "admitted": len(result["mechanically_admitted_closures"])}))


if __name__ == "__main__":
    main()
