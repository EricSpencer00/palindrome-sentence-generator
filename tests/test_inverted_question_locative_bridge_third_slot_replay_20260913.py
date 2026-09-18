"""Tests for actual replay through the third typed lexical slot."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("third_slot_replay_20260913", ROOT / "experiments/inverted_question_locative_bridge_third_slot_replay_20260913.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_actual_replay_reaches_eight_pairs_and_third_slot():
    row = MODULE.replay_bridge(MODULE.ThirdSlotGrammar(6))
    assert row["eight_events_replayed"] and row["third_slot_event"]
    assert row["bridge_pairs"] == [("t", "t"), ("a", "a"), ("t", "t"), ("e", "e"), ("m", "m"), ("p", "p"), ("o", "o"), ("h", "h")]
    assert row["bridge_stream"] == row["terminal_reverse_stream"] == "tatempoh"
    assert row["independent_parse"] and not row["exact_audit"]["exact"]


def test_third_slot_is_typed_compound_location_material():
    grammar = MODULE.ThirdSlotGrammar(1)
    symbol = MODULE.BASE.sym("NP", role="complement", type="location", number="sing", depth="1")
    assert any(p.identifier.endswith(":compound") for p in grammar.productions(symbol))
    text = "was it a tempo hall that a bishop met at i saw"
    tree = MODULE.QUESTION.parse_tree(grammar, text)
    assert tree is not None


def test_long_control_remains_independently_parsed():
    grammar = MODULE.ThirdSlotGrammar(6)
    state = MODULE.SECOND.SECOND.LOCATIVE.explicit_control(grammar); text = MODULE.BASE.render(state)
    row = MODULE.SECOND.SECOND.audit(grammar, text, "control", state.trace)
    assert len(text.replace(" ", "")) >= 100 and row["independent_parse"]
    assert row["feature_witness"]["agreement_ok"] and row["feature_witness"]["valency_ok"]
