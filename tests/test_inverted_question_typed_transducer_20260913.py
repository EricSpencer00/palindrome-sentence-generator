"""Focused tests for the same-tree typed lexical substitution transducer."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("inverted_question_typed_transducer_20260913", ROOT / "experiments/inverted_question_typed_transducer_20260913.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_expanded_grammar_independently_parses_typed_nested_question():
    grammar = MODULE.TransducerGrammar(max_depth=2)
    text = "was it a quiet portrait that a careful teacher that a brave guard greeted praised i saw"
    tree = MODULE.QUESTION.parse_tree(grammar, text)
    assert tree is not None
    witness = MODULE.QUESTION.feature_witness(tree)
    assert witness["relative_count"] == 2
    assert witness["agreement_ok"] and witness["valency_ok"]


def test_shifted_boundary_oracle_uses_letters_not_token_alignment():
    # Right word ends in 'a', satisfying a one-character residual from the
    # left even though the word boundary is asymmetric.
    root = MODULE.BASE.Node(0, MODULE.BASE.Symbol("S", ()), children=(1, 2))
    nodes = (root, MODULE.BASE.Node(1, MODULE.BASE.Symbol("T", ()), terminal="a"), MODULE.BASE.Node(2, MODULE.BASE.Symbol("T", ()), terminal="bba"))
    leaves = (MODULE.BASE.Leaf(1, 1, "a", "word"), MODULE.BASE.Leaf(2, 2, "bba", "word"))
    state = MODULE.BASE.State((1, 2), nodes, leaves, "a", 1, 1, ())
    assert MODULE.boundary_compatible(state, -1)
    assert MODULE.QUESTION.exact_audit("a bba")["shifted_word_boundaries"]


def test_incompatible_residual_is_rejected_at_exposed_lexical_leaf():
    root = MODULE.BASE.Node(0, MODULE.BASE.Symbol("S", ()), children=(1, 2))
    nodes = (root, MODULE.BASE.Node(1, MODULE.BASE.Symbol("T", ()), terminal="a"), MODULE.BASE.Node(2, MODULE.BASE.Symbol("T", ()), terminal="bbc"))
    leaves = (MODULE.BASE.Leaf(1, 1, "a", "word"), MODULE.BASE.Leaf(2, 2, "bbc", "word"))
    state = MODULE.BASE.State((1, 2), nodes, leaves, "a", 1, 1, ())
    assert not MODULE.boundary_compatible(state, -1)


def test_corrupted_trace_does_not_replace_independent_parse():
    grammar = MODULE.TransducerGrammar(max_depth=0)
    row = MODULE.audit(grammar, "was it a quiet portrait i saw", "corrupted_trace", ((999, "fake"),))
    assert row["independent_parse"] and row["feature_witness"]["complete_tree"]
    assert row["shared_tree_trace"][0][1] == "fake"


def test_object_patient_verb_cannot_be_substituted_for_person_gap():
    grammar = MODULE.TransducerGrammar(max_depth=1)
    text = "was it a quiet portrait that a careful teacher greeted i saw"
    assert MODULE.QUESTION.parse_tree(grammar, text) is None


def test_solver_preserves_full_leaf_closure_and_has_no_shortcut_result():
    result = MODULE.solver(MODULE.TransducerGrammar(max_depth=2), max_states=5000)
    assert all(row["independent_parse"] for row in result["exact_closures"])
    assert all(row["independent_exact_audit"]["exact"] for row in result["exact_closures"])


def test_unexposed_np_slots_remain_unlexicalized_before_right_cancellation():
    grammar = MODULE.TransducerGrammar(max_depth=2)
    root = MODULE.BASE.Node(0, grammar.start())
    initial = MODULE.BASE.State((0,), (root,), (), "", 0, 0, ())
    expanded = MODULE.BASE.expand(grammar, initial, 0)[0]
    assert all(leaf.label.startswith("fixed_") for leaf in expanded.leaves)
    after_right = MODULE.BASE.emit(expanded, -1)
    assert after_right is not None and after_right.owner == -1
    after_left = MODULE.BASE.emit(after_right, 1)
    assert after_left is not None and after_left.residual == ""
    assert all(leaf.label.startswith("fixed_") for leaf in after_left.leaves)
