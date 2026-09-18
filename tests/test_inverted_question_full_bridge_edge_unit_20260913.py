"""Tests for the full two-character envelope bridge wrapper."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "full_bridge_edge_unit_20260913",
    ROOT / "experiments/inverted_question_full_bridge_edge_unit_20260913.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_full_bridge_requires_and_finds_at_not_just_final_t():
    bridge = MODULE.full_boundary_bridge()
    assert bridge["fixed_prefix_after_wasi"] == "t"
    assert bridge["required_core_suffix"] == "at"
    assert bridge["typed_object_verbs_matching_full_at"] == ("beat",)
    assert bridge["satisfiable"]


def test_bridge_verb_is_a_real_typed_grammar_leaf():
    grammar = MODULE.FullBridgeGrammar(0)
    symbol = MODULE.BASE.sym("LEX_SLOT", category="verb", patient_type="object", role="relative_verb")
    assert any(p.identifier == "LEX_SLOT:beat" for p in grammar.productions(symbol))
    text = "was it a portrait i saw"
    row = MODULE.audit(grammar, text, "reparse", ())
    assert row["independent_parse"] and row["feature_witness"]["valency_ok"]


def test_long_control_is_independently_parsed_and_not_claimed_as_palindrome():
    grammar = MODULE.FullBridgeGrammar(6)
    state = MODULE.EDGE.explicit_control(grammar)
    text = MODULE.BASE.render(state)
    row = MODULE.audit(grammar, text, "control", state.trace)
    words = [x.word for x in MODULE.BASE.ordered_leaves(state) if x.label not in MODULE.FIXED_LABELS]
    assert len(text.replace(" ", "")) >= 100
    assert row["independent_parse"] and row["feature_witness"]["agreement_ok"]
    assert row["feature_witness"]["valency_ok"] and len(words) == len(set(words))
    assert not row["independent_exact_audit"]["exact"]

