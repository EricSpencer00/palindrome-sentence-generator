"""Tests for the internal-character centre discourse operator."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "internal_character_center_discourse_20260913",
    ROOT / "experiments/internal_character_center_discourse_20260913.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_center_is_a_single_word_internal_pivot_not_a_multiword_seed():
    grammar = MODULE.Grammar()
    leaves = MODULE.slots(grammar.expand(MODULE.Symbol("S")))
    assert leaves[5].symbol.role == "center_word"
    assert MODULE.normalize_letters("te") == "te"
    assert MODULE.normalize_letters("eth") == "eth"
    assert MODULE.forbidden_spans(("teeth",)) == []


def test_internal_pivot_records_first_literal_mismatch_and_replay():
    trace = MODULE.internal_trace("teeth", "examined", "during")
    assert not trace["completed"]
    assert trace["first_incompatible"]["char"] == "h"
    assert trace["first_incompatible"]["expected"] == "d"
    assert trace["replay"]["events_replayed"] == 6
    assert trace["replay"]["cancellations"] == 2
    assert all(len(event["char"]) == 1 for event in trace["ledger"])


def test_fresh_run_uses_shared_referent_controls_and_no_candidate():
    result = MODULE.run(state_limit=100_000, closure_limit=100)
    assert result["config"]["word_internal_center_pivot"]
    assert result["config"]["two_sentence_shared_referent"]
    assert result["discovery"]["status"] == "first_literal_mismatch"
    assert result["exact_closures"] == []
    assert result["admitted_closures"] == []
    assert all(row["independent_parse"] for row in result["complete_grammar_controls"])
    assert all(row["independent_exact_audit"]["letters"] > 100 for row in result["complete_grammar_controls"])
    assert all(row["rendered"].count(".") == 2 for row in result["complete_grammar_controls"])

