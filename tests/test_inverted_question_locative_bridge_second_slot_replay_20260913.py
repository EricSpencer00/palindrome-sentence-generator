"""Tests for executable BASE.emit replay of the second-slot bridge."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("second_slot_replay_20260913", ROOT / "experiments/inverted_question_locative_bridge_second_slot_replay_20260913.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_replay_emits_six_bridge_pairs_from_actual_tree():
    row = MODULE.replay_bridge(MODULE.SECOND.SecondSlotGrammar(6))
    assert row["six_events_replayed"] and row["second_slot_event"]
    assert row["bridge_pairs"] == [("t", "t"), ("a", "a"), ("t", "t"), ("e", "e"), ("m", "m"), ("p", "p")]
    assert row["bridge_stream"] == row["terminal_reverse_stream"] == "tatemp"
    assert row["independent_parse"] and not row["exact_audit"]["exact"]


def test_replay_events_are_not_hardcoded_slots():
    row = MODULE.replay_bridge(MODULE.SECOND.SecondSlotGrammar(6))
    assert [event["word"] for event in row["emitter_events"][-4:]] == ["temple", "met", "temple", "bishop"]
    assert [event["slot"] for event in row["emitter_events"][-2:]] == ["noun_complement", "noun_relative_agent"]
    assert row["emitter_events"][-1]["character"] == "p"


def test_long_control_remains_independently_parsed():
    grammar = MODULE.SECOND.SecondSlotGrammar(6)
    state = MODULE.SECOND.LOCATIVE.explicit_control(grammar); text = MODULE.BASE.render(state)
    row = MODULE.SECOND.audit(grammar, text, "control", state.trace)
    assert len(text.replace(" ", "")) >= 100 and row["independent_parse"]
    assert row["feature_witness"]["agreement_ok"] and row["feature_witness"]["valency_ok"]
