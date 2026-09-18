"""Tests for the independent typed edge-unit tree construction."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "edge_unit_tree_20260913",
    ROOT / "experiments/inverted_question_edge_unit_tree_20260913.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_outer_seam_feasibility_has_authored_t_ending_object_verb():
    assert "built" in MODULE.boundary_feasibility("verb", -1, "t")
    grammar = MODULE.EdgeUnitGrammar(0)
    assert any(p.identifier == "LEX_SLOT:built"
               for p in grammar.productions(MODULE.BASE.sym("LEX_SLOT", category="verb",
                                                              patient_type="object", role="relative_verb")))


def test_live_debt_reads_the_active_morpheme_unit_across_affix_progress():
    lexeme = MODULE.LEXEME_BY_FORM["trusted"]
    first = MODULE.BASE.Leaf(7, 7, "trusted", "relative_verb")
    assert MODULE.active_unit(lexeme, first, -1) == "ed"
    after_d = MODULE.BASE.Leaf(7, 7, "trusted", "relative_verb", right=1)
    assert MODULE.active_unit(lexeme, after_d, -1) == "ed"
    after_ed = MODULE.BASE.Leaf(7, 7, "trusted", "relative_verb", right=2)
    assert MODULE.active_unit(lexeme, after_ed, -1) == "trust"


def test_shifted_boundary_edge_filter_uses_live_unit_not_word_boundary():
    root = MODULE.BASE.Node(0, MODULE.BASE.Symbol("S", ()), children=(1, 2))
    nodes = (root,
             MODULE.BASE.Node(1, MODULE.BASE.Symbol("T", ()), terminal="a"),
             MODULE.BASE.Node(2, MODULE.BASE.Symbol("T", ()), terminal="trusted"))
    leaves = (MODULE.BASE.Leaf(1, 1, "a", "word"),
              MODULE.BASE.Leaf(2, 2, "trusted", "relative_verb"))
    state = MODULE.BASE.State((1, 2), nodes, leaves, "d", 1, 1, ())
    assert MODULE.lexical_unit_accepts(state, -1, [leaves[1]])
    assert MODULE.QUESTION.exact_audit("a trusted")["shifted_word_boundaries"]


def test_control_is_long_independently_reparsed_and_nonrepetitive():
    grammar = MODULE.EdgeUnitGrammar(6)
    state = MODULE.explicit_control(grammar)
    text = MODULE.BASE.render(state)
    row = MODULE.audit(grammar, text, "test_edge_unit_control", state.trace)
    words = [x.word for x in MODULE.BASE.ordered_leaves(state) if x.label not in MODULE.FIXED_LABELS]
    assert len(text.replace(" ", "")) >= 100
    assert row["independent_parse"] and row["feature_witness"]["complete_tree"]
    assert row["feature_witness"]["agreement_ok"] and row["feature_witness"]["valency_ok"]
    assert row["content_word_forms_unique"] if "content_word_forms_unique" in row else len(words) == len(set(words))
    assert not row["independent_exact_audit"]["exact"]


def test_corrupted_trace_does_not_replace_independent_reparse():
    grammar = MODULE.EdgeUnitGrammar(0)
    row = MODULE.audit(grammar, "was it a portrait i saw", "corrupted", ((999, "fake"),))
    assert row["independent_parse"] and row["feature_witness"]["complete_tree"]
    assert row["shared_tree_trace"] == [(999, "fake")]
