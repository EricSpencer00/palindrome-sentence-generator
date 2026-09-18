"""Independent invariants for the single-shared-tree intersection."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cfg_feature_palindrome_intersection_20260913", ROOT / "experiments/cfg_feature_palindrome_intersection_20260913.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def tiny():
    return MODULE.FeatureGrammar(max_clauses=1)


def test_tiny_exhaustive_oracle_and_full_feature_witness():
    grammar = tiny()
    # The oracle checks the finite top-level branch count, then a complete
    # typed derivation. A one-letter palindromic fragment has no tree witness.
    assert len(grammar.productions(grammar.start())) == 1
    assert MODULE.parse_tree(grammar, "a") is None
    text = "a artist guides a artist"
    tree = MODULE.parse_tree(grammar, text)
    assert tree is not None
    witness = MODULE.features(tree)
    assert witness["complete_tree"] and witness["agreement_ok"] and witness["valency_ok"]
    assert witness["semantic_roles"]
    assert witness["events"] == ["guide"]


def test_shifted_word_boundaries_are_not_used_for_character_cancellation():
    audit = MODULE.exact_audit("a bba")
    assert audit["exact"]
    assert audit["shifted_word_boundaries"]


def test_corrupted_trace_cannot_replace_independent_reparse():
    grammar = tiny()
    row = MODULE.audit(grammar, "a artist guides a artist", "corrupted_trace", ((999, "not-a-real-production"),))
    assert row["independent_parse"]
    assert row["feature_witness"]["complete_tree"]
    assert row["shared_tree_trace"][0][1] == "not-a-real-production"


def test_agreement_and_valency_impossible_cases_fail_complete_reparse():
    grammar = tiny()
    assert MODULE.parse_tree(grammar, "the artists guides a artist") is None
    assert MODULE.parse_tree(grammar, "a artist carries a artist") is None


def test_independent_reparse_backtracks_past_single_clause_to_coordination():
    grammar = MODULE.FeatureGrammar(max_clauses=2)
    text = "a artist guides a artist and a artist guides a artist"
    tree = MODULE.parse_tree(grammar, text)
    assert tree is not None
    assert MODULE.features(tree)["events"] == ["guide", "guide"]


def test_exhaustive_tiny_tree_oracle_compares_every_surface_to_fresh_exact_audit():
    class OracleGrammar(MODULE.FeatureGrammar):
        def productions(self, lhs):
            rows = super().productions(lhs)
            if lhs.name == "CLAUSE":
                rows = tuple(x for x in rows if x.identifier == "CLAUSE:guide:sing")
            elif lhs.name == "VP":
                rows = tuple(x for x in rows if x.identifier.endswith(":bare"))
            elif lhs.name == "NP":
                rows = tuple(x for x in rows if x.identifier.endswith("::artist"))
            return rows

    grammar = OracleGrammar(max_clauses=1)
    root = MODULE.Node(0, grammar.start())
    pending = [MODULE.State((0,), (root,), (), "", 0, 0, ())]
    expanded = []
    while pending:
        state = pending.pop()
        nodes = MODULE.node_map(state)
        unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        if not unresolved:
            expanded.append(state)
            continue
        pending.extend(MODULE.expand(grammar, state, unresolved[0]))
    assert len(expanded) == 1
    for state in expanded:
        text = MODULE.render(state)
        audit = MODULE.exact_audit(text)
        assert audit["letters"] == sum(len(word) for word in text.split())
        assert MODULE.parse_tree(grammar, text) is not None


def test_intersection_closures_equal_fresh_exact_audits_of_all_tiny_derivations():
    left = "ab" * 25
    right = left[::-1]

    class PalOracle(MODULE.FeatureGrammar):
        def __init__(self):
            super().__init__(1)
        def productions(self, lhs):
            if lhs.name == "S":
                return (MODULE.Production("S:oracle", lhs, (MODULE.sym("CLAUSE"),)),)
            if lhs.name == "CLAUSE":
                return (
                    MODULE.Production("CLAUSE:pal", lhs, tuple(MODULE.sym("T", label="X", form=x) for x in (left, "c", right))),
                    MODULE.Production("CLAUSE:nonpal", lhs, tuple(MODULE.sym("T", label="X", form=x) for x in (left, "c", "bb" * 25))),
                )
            return super().productions(lhs)

    grammar = PalOracle()
    root = MODULE.Node(0, grammar.start())
    states = [MODULE.State((0,), (root,), (), "", 0, 0, ())]
    complete_surfaces = []
    while states:
        state = states.pop()
        nodes = MODULE.node_map(state)
        unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        if unresolved:
            states.extend(MODULE.expand(grammar, state, unresolved[0]))
        else:
            complete_surfaces.append(MODULE.render(state))
    expected = {text for text in complete_surfaces if MODULE.exact_audit(text)["exact"]}
    result = MODULE.intersect(grammar, max_states=1000)
    observed = {row["rendered"] for row in result["exact_closures"]}
    assert observed == expected
    assert len(observed) == 1
    assert result["exact_closures"][0]["independent_parse"]


def test_no_shorthand_grammar_witness_for_long_palindromic_fragment():
    grammar = tiny()
    text = "z" * 120
    row = MODULE.audit(grammar, text, "fragment", ())
    assert row["independent_exact_audit"]["exact"]
    assert not row["independent_parse"]
    assert "independent_complete_reparse_failed" in row["rejection_codes"]


def test_shared_tree_frontier_expands_without_stitching_stacks():
    grammar = tiny()
    root = MODULE.Node(0, grammar.start())
    state = MODULE.State((0,), (root,), (), "", 0, 0, ())
    expanded = MODULE.expand(grammar, state, 0)
    assert expanded
    assert all(item.frontier != (0,) for item in expanded)
    assert all(len(item.trace) == 1 for item in expanded)


def test_multi_character_residual_keeps_original_owner_until_drained():
    root = MODULE.Node(0, MODULE.Symbol("S", ()), children=(1, 2))
    nodes = (root, MODULE.Node(1, MODULE.Symbol("T", ()), terminal="abcd"), MODULE.Node(2, MODULE.Symbol("T", ()), terminal="dcba"))
    leaves = (MODULE.Leaf(1, 1, "abcd", "word"), MODULE.Leaf(2, 2, "dcba", "word"))
    state = MODULE.State((1, 2), nodes, leaves, "", 0, 0, ())
    state = MODULE.emit(state, 1)
    state = MODULE.emit(state, 1)
    assert state is not None and state.residual == "ab" and state.owner == 1
    state = MODULE.emit(state, -1)
    assert state is not None and state.residual == "b" and state.owner == 1
    state = MODULE.emit(state, -1)
    assert state is not None and state.residual == "" and state.owner == 0
