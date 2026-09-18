"""Tests for the complete terminal-to-prefix bridge preflight."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "full_bridge_preflight_20260913",
    ROOT / "experiments/inverted_question_full_bridge_preflight_20260913.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_preflight_simulates_full_first_word_not_one_seam_character():
    row = MODULE.bridge_preflight(MODULE.PreflightGrammar(0))
    assert row["excluded_negative_oracle"]
    assert row["left_bridge_stream"] == row["right_terminal_reverse_stream"] == "tab"
    assert row["seam_pairs"] == [("t", "t"), ("a", "a"), ("b", "b")]
    assert row["full_multi_character_match"] and row["independent_parse"]
    assert row["exact_audit"]["exact"]


def test_oracle_word_is_not_search_candidate_evidence():
    row = MODULE.bridge_preflight(MODULE.PreflightGrammar(0))
    assert row["rendered"] not in [x["rendered"] for x in MODULE.run(max_states=1)["exact_closures"]]
    assert row["reason"].startswith("known short palindrome used only")


def test_recursive_control_remains_long_and_independently_parsed():
    grammar = MODULE.PreflightGrammar(6)
    state = MODULE.FULL.EDGE.explicit_control(grammar)
    text = MODULE.BASE.render(state)
    row = MODULE.audit(grammar, text, "control", state.trace)
    content = [x.word for x in MODULE.BASE.ordered_leaves(state) if x.label not in MODULE.FIXED_LABELS]
    assert len(text.replace(" ", "")) >= 100
    assert row["independent_parse"] and row["feature_witness"]["agreement_ok"]
    assert row["feature_witness"]["valency_ok"] and len(content) == len(set(content))
    assert not row["independent_exact_audit"]["exact"]

