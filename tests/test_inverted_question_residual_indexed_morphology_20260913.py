"""Tests for residual-indexed typed morphology."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("residual_indexed_morphology_20260913", ROOT / "experiments/inverted_question_residual_indexed_morphology_20260913.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_unmodified_np_and_nested_relative_reparse():
    grammar = MODULE.ResidualIndexedGrammar(2)
    text = "was it a portrait that a teacher that a guard greeted praised i saw"
    tree = MODULE.QUESTION.parse_tree(grammar, text)
    assert tree is not None
    witness = MODULE.QUESTION.feature_witness(tree)
    assert witness["relative_count"] == 2 and witness["agreement_ok"] and witness["valency_ok"]


def test_signature_index_has_first_and_last_side_keys_for_typed_forms():
    assert MODULE.SIGNATURE_INDEX[("noun", 1, "p")]
    assert MODULE.SIGNATURE_INDEX[("noun", -1, "t")]
    assert MODULE.SIGNATURE_INDEX[("verb", -1, "d")]


def test_shifted_boundary_oracle_uses_side_signature_not_word_boundary():
    root = MODULE.BASE.Node(0, MODULE.BASE.Symbol("S", ()), children=(1, 2))
    nodes = (root, MODULE.BASE.Node(1, MODULE.BASE.Symbol("T", ()), terminal="a"), MODULE.BASE.Node(2, MODULE.BASE.Symbol("T", ()), terminal="bba"))
    leaves = (MODULE.BASE.Leaf(1, 1, "a", "word"), MODULE.BASE.Leaf(2, 2, "bba", "word"))
    state = MODULE.BASE.State((1, 2), nodes, leaves, "t", 1, 1, ())
    indexed_leaf = MODULE.BASE.Leaf(2, 2, "portrait", "noun_test")
    assert MODULE.residual_index_accepts(state, -1, [indexed_leaf])
    assert MODULE.QUESTION.exact_audit("a bba")["shifted_word_boundaries"]


def test_mismatched_live_signature_is_rejected():
    root = MODULE.BASE.Node(0, MODULE.BASE.Symbol("S", ()), children=(1, 2))
    nodes = (root, MODULE.BASE.Node(1, MODULE.BASE.Symbol("T", ()), terminal="a"), MODULE.BASE.Node(2, MODULE.BASE.Symbol("T", ()), terminal="bbc"))
    leaves = (MODULE.BASE.Leaf(1, 1, "a", "word"), MODULE.BASE.Leaf(2, 2, "bbc", "word"))
    state = MODULE.BASE.State((1, 2), nodes, leaves, "t", 1, 1, ())
    mismatched_leaf = MODULE.BASE.Leaf(2, 2, "bbc", "noun_test")
    assert not MODULE.residual_index_accepts(state, -1, [mismatched_leaf])


def test_corrupted_trace_is_not_a_reparse_witness():
    grammar = MODULE.ResidualIndexedGrammar(0)
    row = MODULE.audit(grammar, "was it a portrait i saw", "corrupted", ((999, "fake"),))
    assert row["independent_parse"] and row["feature_witness"]["complete_tree"]
    assert row["shared_tree_trace"][0][1] == "fake"


def test_solver_has_no_shortcut_outputs_and_requires_full_leaf_closure():
    result = MODULE.solver(MODULE.ResidualIndexedGrammar(2), max_states=3000)
    assert all(row["independent_parse"] for row in result["exact_closures"])
    assert all(row["independent_exact_audit"]["exact"] for row in result["exact_closures"])


def test_unexposed_np_slots_are_not_lexicalized_before_opposite_edge_progress():
    grammar = MODULE.ResidualIndexedGrammar(2)
    root = MODULE.BASE.Node(0, grammar.start())
    initial = MODULE.BASE.State((0,), (root,), (), "", 0, 0, ())
    expanded = MODULE.BASE.expand(grammar, initial, 0)[0]
    assert all(leaf.label.startswith("fixed_") for leaf in expanded.leaves)
    after_right = MODULE.BASE.emit(expanded, -1)
    assert after_right is not None and after_right.owner == -1
    after_left = MODULE.BASE.emit(after_right, 1)
    assert after_left is not None and after_left.residual == ""
    assert all(leaf.label.startswith("fixed_") for leaf in after_left.leaves)


def test_explicit_long_control_is_complete_reparsed_and_content_unique():
    grammar = MODULE.ResidualIndexedGrammar(6)
    state = MODULE.explicit_control(grammar)
    text = MODULE.BASE.render(state)
    row = MODULE.audit(grammar, text, "test_control", state.trace)
    content = [leaf.word for leaf in MODULE.BASE.ordered_leaves(state)
               if leaf.label not in MODULE.FIXED_LABELS]
    assert row["independent_parse"]
    assert row["feature_witness"]["complete_tree"]
    assert row["feature_witness"]["agreement_ok"] and row["feature_witness"]["valency_ok"]
    assert row["independent_exact_audit"]["letters"] >= 100
    assert not row["independent_exact_audit"]["exact"]
    assert len(content) == len(set(content))
