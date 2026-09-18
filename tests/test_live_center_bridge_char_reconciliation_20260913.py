"""Tests for the live, character-at-a-time center bridge."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("live_center_bridge_char_reconciliation_20260913", ROOT / "experiments/live_center_bridge_char_reconciliation_20260913.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_semantic_center_pair_is_not_a_reverse_word_pair():
    assert MODULE.LICENSED_COMPOUNDS[("paper", "report")] == "a report written on paper"
    assert "paper"[::-1] != "report"
    tree = MODULE.Grammar().expand(MODULE.Grammar().start())
    leaves = MODULE.slots(tree)
    assert leaves[7].symbol.role == "compound_modifier"
    assert leaves[8].symbol.role == "compound_head"


def test_bridge_trace_alternates_every_character_and_records_three_cancellations():
    trace = MODULE.bridge_trace("paper", "report")
    events = trace["trace"]
    assert [(x["side"], x["char"]) for x in events[:6]] == [("left", "r"), ("right", "r"), ("left", "e"), ("right", "e"), ("left", "p"), ("right", "p")]
    assert trace["stats"]["cancellations"] == 3
    assert events[-1]["action"] == "contradiction"
    assert not trace["completed"]


def test_fresh_100k_run_has_complete_parsed_controls_and_no_candidate():
    result = MODULE.run(state_limit=100_000, closure_limit=100)
    assert result["exact_closures"] == []
    assert result["config"]["one_character_emission_states"]
    assert all(row["independent_parse"] for row in result["complete_grammar_controls"])
    assert all(row["independent_exact_audit"]["letters"] > 100 for row in result["complete_grammar_controls"])
    assert result["stats"]["center_bridge_contradictions"] == 1


def test_partial_bridge_never_bypasses_full_tree_reparse():
    grammar = MODULE.Grammar()
    row = MODULE.audit(grammar, "Paper report.", "center_bridge_fragment", ("paper", "report"))
    assert not row["independent_parse"]
    assert not row["mechanically_admitted"]

