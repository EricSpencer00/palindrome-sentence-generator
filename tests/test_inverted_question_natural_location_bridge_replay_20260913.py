"""Tests for the natural temple-hall bridge."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("natural_location_bridge_20260913", ROOT / "experiments/inverted_question_natural_location_bridge_replay_20260913.py")
MODULE = importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; sys.modules[SPEC.name] = MODULE; SPEC.loader.exec_module(MODULE)


def test_natural_compound_replay_reaches_nine_pairs():
    row = MODULE.replay_bridge(MODULE.NaturalLocationGrammar(6))
    assert row["nine_pairs_replayed"] and row["third_slot_event"]
    assert row["bridge_stream"] == row["terminal_reverse_stream"] == "tatempleh"
    assert row["independent_parse"] and not row["exact_audit"]["exact"]


def test_compound_and_collective_subject_are_typed():
    grammar = MODULE.NaturalLocationGrammar(1)
    tree = MODULE.QUESTION.parse_tree(grammar, "was it a temple hall that the help met at i saw")
    witness = MODULE.feature_witness(tree)
    assert tree is not None and witness["agreement_ok"] and witness["valency_ok"]
    assert any(role.get("type") == "person_group" for role in witness["semantic_roles"])


def test_long_control_is_independently_parsed():
    grammar = MODULE.NaturalLocationGrammar(6); state = MODULE.SECOND.SECOND.LOCATIVE.explicit_control(grammar); text = MODULE.BASE.render(state)
    row = MODULE.audit(grammar, text, "control", state.trace)
    assert len(text.replace(" ", "")) >= 100 and row["independent_parse"]
    assert row["feature_witness"]["agreement_ok"] and row["feature_witness"]["valency_ok"]

