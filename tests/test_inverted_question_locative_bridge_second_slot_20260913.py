"""Tests for the second typed lexical slot bridge."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("locative_second_slot_20260913", ROOT / "experiments/inverted_question_locative_bridge_second_slot_20260913.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_bridge_reaches_second_typed_slot_beyond_tatem():
    row = MODULE.bridge_preflight(MODULE.SecondSlotGrammar(6))
    assert row["bridge_stream"] == row["terminal_reverse_stream"] == "tatemp"
    assert row["full_multi_character_match"] and row["second_typed_slot_reached"]
    assert row["live_emitter_trace"][5]["right_slot"] == "relative_agent_noun_bishop"
    assert row["independent_parse"] and not row["exact_audit"]["exact"]


def test_bishop_is_an_authored_person_lexical_leaf():
    grammar = MODULE.SecondSlotGrammar(1)
    symbol = MODULE.BASE.sym("LEX_SLOT", category="noun", type="person", role="relative_agent", number="sing")
    assert any(p.identifier == "LEX_SLOT:bishop" for p in grammar.productions(symbol))


def test_long_control_is_independently_parsed_and_unique():
    grammar = MODULE.SecondSlotGrammar(6)
    state = MODULE.LOCATIVE.explicit_control(grammar); text = MODULE.BASE.render(state)
    row = MODULE.audit(grammar, text, "control", state.trace)
    content = [x.word for x in MODULE.BASE.ordered_leaves(state) if x.label not in MODULE.FIXED_LABELS]
    assert len(text.replace(" ", "")) >= 100
    assert row["independent_parse"] and row["feature_witness"]["agreement_ok"] and row["feature_witness"]["valency_ok"]
    assert len(content) == len(set(content)) and not row["independent_exact_audit"]["exact"]

