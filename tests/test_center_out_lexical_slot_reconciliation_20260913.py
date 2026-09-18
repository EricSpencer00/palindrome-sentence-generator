"""Invariants for center-out lexical slot reconciliation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "center_out_lexical_slot_reconciliation_20260913",
    ROOT / "experiments/center_out_lexical_slot_reconciliation_20260913.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_connected_tree_owns_all_twenty_terminal_leaves():
    grammar = MODULE.FeatureGrammar()
    tree = grammar.expand(grammar.start())
    leaves = MODULE.slots(tree)
    assert tree.symbol.name == "S"
    assert len(leaves) == 20
    assert all(slot.symbol.name == "T" for slot in leaves)
    assert leaves[9].symbol.role == "event_adj"
    assert leaves[10].symbol.role == "event"


def test_center_boundary_feasibility_is_explicit_before_search():
    grammar = MODULE.FeatureGrammar()
    feasibility = MODULE.center_boundary_feasibility(MODULE.slots(grammar.expand(grammar.start())))
    assert feasibility["left_role"] == "event_adj"
    assert feasibility["right_role"] == "event"
    assert feasibility["compatible_pair_count"] == 0


def test_center_out_character_reconciliation_is_not_word_boundary_matching():
    stats = MODULE.Counter()
    left = MODULE.emit_left_outward("", "", "ab", stats)
    assert left == ("ba", "")
    right = MODULE.emit_right_outward(left[0], left[1], "ba", stats)
    assert right == ("", "")
    assert stats["residual_cancellations"] == 2


def test_controls_are_complete_parses_over_one_tree_and_over_target_length():
    result = MODULE.run(state_limit=100_000, closure_limit=100)
    assert result["exact_closures"] == []
    assert all(row["independent_parse"] for row in result["complete_grammar_controls"])
    assert all(row["independent_exact_audit"]["letters"] > 100 for row in result["complete_grammar_controls"])
    assert result["center_boundary_feasibility"]["compatible_pair_count"] == 0

