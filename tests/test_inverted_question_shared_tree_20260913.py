"""Tests for the root-specific inverted-question construction."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("inverted_question_shared_tree_20260913", ROOT / "experiments/inverted_question_shared_tree_20260913.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_fixed_envelope_is_independently_parsed_as_one_question_tree():
    grammar = MODULE.InvertedQuestionGrammar(max_depth=0)
    text = "was it a quiet portrait i saw"
    tree = MODULE.parse_tree(grammar, text)
    assert tree is not None
    witness = MODULE.feature_witness(tree)
    assert witness["complete_tree"] and witness["agreement_ok"] and witness["valency_ok"]


def test_recursive_typed_relative_is_parsed_with_roles_and_valency():
    grammar = MODULE.InvertedQuestionGrammar(max_depth=1)
    text = "was it a quiet portrait that a careful teacher praised i saw"
    tree = MODULE.parse_tree(grammar, text)
    assert tree is not None
    witness = MODULE.feature_witness(tree)
    assert witness["relative_count"] == 1
    assert witness["semantic_roles"]
    assert witness["agreement_ok"] and witness["valency_ok"]


def test_genuinely_nested_person_relative_is_independently_parsed():
    grammar = MODULE.InvertedQuestionGrammar(max_depth=2)
    text = "was it a quiet portrait that a careful teacher that a brave guard greeted praised i saw"
    tree = MODULE.parse_tree(grammar, text)
    assert tree is not None
    witness = MODULE.feature_witness(tree)
    assert witness["relative_count"] == 2
    assert witness["agreement_ok"] and witness["valency_ok"]


def test_long_recursive_control_uses_distinct_authored_content_words():
    grammar = MODULE.InvertedQuestionGrammar(max_depth=6)
    state = MODULE.explicit_recursive_control(grammar)
    words = MODULE.BASE.render(state).split()
    assert len("".join(words)) >= 100
    function_words = {"was", "it", "that", "a", "the", "i", "saw"}
    content = [word for word in words if word not in function_words]
    assert len(content) == len(set(content))


def test_agreement_and_valency_impossible_surfaces_fail_independent_reparse():
    grammar = MODULE.InvertedQuestionGrammar(max_depth=1)
    assert MODULE.parse_tree(grammar, "was it a quiet portrait that a careful teacher praise i saw") is None
    assert MODULE.parse_tree(grammar, "was it a quiet portrait that a careful teacher greeted i saw") is None


def test_corrupted_shared_tree_trace_does_not_substitute_for_reparse():
    grammar = MODULE.InvertedQuestionGrammar(max_depth=0)
    row = MODULE.audit(grammar, "was it a quiet portrait i saw", "corrupted_trace", ((999, "fake"),))
    assert row["independent_parse"]
    assert row["feature_witness"]["complete_tree"]
    assert row["shared_tree_trace"][0][1] == "fake"


def test_shifted_word_boundaries_are_reported_independently_of_character_tape():
    audit = MODULE.exact_audit("a bba")
    assert audit["exact"] and audit["shifted_word_boundaries"]


def test_single_shared_frontier_has_no_separate_suffix_stack():
    grammar = MODULE.InvertedQuestionGrammar(max_depth=1)
    root = MODULE.BASE.Node(0, grammar.start())
    state = MODULE.BASE.State((0,), (root,), (), "", 0, 0, ())
    branches = MODULE.BASE.expand(grammar, state, 0)
    assert len(branches) == 1
    assert all(len(branch.trace) == 1 for branch in branches)
    assert all(len(branch.frontier) == 5 for branch in branches)


def test_root_specific_exact_closure_oracle_matches_fresh_audits():
    # Synthetic oracle only: this is not a known/canonical palindrome and is
    # never promoted as an English result. Its interior is a fresh palindrome
    # plus the required residual 't', so the fixed envelope closes exactly.
    interior = ("abccba" * 15) + "t"

    class Oracle(MODULE.InvertedQuestionGrammar):
        def __init__(self):
            super().__init__(0)
        def productions(self, lhs):
            if lhs.name == "NP":
                return (MODULE.BASE.Production("NP:oracle", lhs,
                    (MODULE.BASE.sym("T", label="oracle_interior", form=interior),)),)
            return super().productions(lhs)

    grammar = Oracle()
    root = MODULE.BASE.Node(0, grammar.start())
    pending = [MODULE.BASE.State((0,), (root,), (), "", 0, 0, ())]
    all_surfaces = []
    while pending:
        state = pending.pop()
        nodes = MODULE.BASE.node_map(state)
        unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        if unresolved:
            pending.extend(MODULE.BASE.expand(grammar, state, unresolved[0]))
        else:
            all_surfaces.append(MODULE.BASE.render(state))
    expected = {text for text in all_surfaces if MODULE.exact_audit(text)["exact"]}
    assert len(expected) == 1
    assert MODULE.exact_audit(next(iter(expected)))["letters"] == 100
    result = MODULE.solver(grammar, max_states=5000)
    observed = {row["rendered"] for row in result["exact_closures"]}
    assert observed == expected
    assert all(row["independent_parse"] for row in result["exact_closures"])
