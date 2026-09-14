"""Tests for the post-nine mismatch capture and entrance grammar."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("entrance_replay_20260913", ROOT / "experiments/inverted_question_natural_location_entrance_replay_20260913.py")
MODULE = importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; sys.modules[SPEC.name] = MODULE; SPEC.loader.exec_module(MODULE)


def test_actual_replay_crosses_compound_and_captures_next_mismatch():
    row = MODULE.replay_bridge(MODULE.EntranceGrammar(6))
    assert row["nine_pairs_replayed"] and row["compound_boundary_crossed"]
    assert row["bridge_stream"] == row["terminal_reverse_stream"] == "tatempleh"
    mismatch = row["first_post_bridge_attempt"]
    assert mismatch["left_character"] == "a" and mismatch["right_character"] == "e"
    assert mismatch["emitter_rejected"]
    assert row["independent_parse"] and not row["exact_audit"]["exact"]


def test_entrance_is_the_licensed_location_head():
    grammar = MODULE.EntranceGrammar(1)
    text = "was it a temple hall entrance that the help met at i saw"
    tree = MODULE.QUESTION.parse_tree(grammar, text)
    witness = MODULE.NATURAL.feature_witness(tree)
    assert tree is not None and witness["agreement_ok"] and witness["valency_ok"]


def test_long_control_is_independently_parsed():
    grammar = MODULE.EntranceGrammar(6); state = MODULE.NATURAL.SECOND.SECOND.LOCATIVE.explicit_control(grammar); text = MODULE.BASE.render(state)
    row = MODULE.NATURAL.audit(grammar, text, "control", state.trace)
    assert len(text.replace(" ", "")) >= 100 and row["independent_parse"]
    assert row["feature_witness"]["agreement_ok"] and row["feature_witness"]["valency_ok"]
