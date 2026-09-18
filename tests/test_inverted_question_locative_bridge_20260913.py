"""Tests for the original locative full-bridge construction."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("locative_bridge_20260913", ROOT / "experiments/inverted_question_locative_bridge_20260913.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_original_bridge_matches_five_characters_through_temple():
    row = MODULE.bridge_preflight(MODULE.LocativeBridgeGrammar(1))
    assert row["bridge_stream"] == row["terminal_reverse_stream"] == "tatem"
    assert row["full_multi_character_match"] and row["independent_parse"]
    assert not row["exact_audit"]["exact"]


def test_locative_relative_has_explicit_at_and_typed_gap():
    grammar = MODULE.LocativeBridgeGrammar(1)
    text = "was it a temple that a teacher met at i saw"
    tree = MODULE.QUESTION.parse_tree(grammar, text)
    witness = MODULE.feature_witness(tree)
    assert tree is not None and witness["relative_count"] == 1
    assert witness["agreement_ok"] and witness["valency_ok"]


def test_long_control_is_independently_parsed_and_content_unique():
    grammar = MODULE.LocativeBridgeGrammar(6)
    state = MODULE.explicit_control(grammar); text = MODULE.BASE.render(state)
    row = MODULE.audit(grammar, text, "control", state.trace)
    content = [x.word for x in MODULE.BASE.ordered_leaves(state) if x.label not in MODULE.FIXED_LABELS]
    assert len(text.replace(" ", "")) >= 100
    assert row["independent_parse"] and row["feature_witness"]["agreement_ok"] and row["feature_witness"]["valency_ok"]
    assert len(content) == len(set(content)) and not row["independent_exact_audit"]["exact"]
