"""Invariants for the connected compound-tree center bridge."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("center_bridge_compound_tree_20260913", ROOT / "experiments/center_bridge_compound_tree_20260913.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_single_tree_has_a_real_four_character_center_bridge():
    grammar = MODULE.CompoundFeatureGrammar()
    leaves = MODULE.slots(grammar.expand(grammar.start()))
    assert len(leaves) == 16
    assert leaves[7].symbol.role == "compound_modifier"
    assert leaves[8].symbol.role == "compound_head"
    bridge = MODULE.center_bridge_examples(leaves)
    assert bridge["bridge_count"] >= 1
    assert any(x["left"] == "part" and x["right"] == "trap" and x["bridge_letters"] == 4 and x["cancellations"] == 4 for x in bridge["compatible_multi_character_bridges"])


def test_center_bridge_reconciles_each_character():
    stats = MODULE.Counter()
    left = MODULE.emit_left_outward("", "", "part", stats)
    assert left == ("trap", "")
    right = MODULE.emit_right_outward(left[0], left[1], "trap", stats)
    assert right == ("", "")
    assert stats["residual_cancellations"] == 4


def test_fresh_100k_run_has_complete_parsed_controls_and_no_candidate():
    result = MODULE.run(state_limit=100_000, closure_limit=100)
    assert result["exact_closures"] == []
    assert result["config"]["multi_character_center_bridge"]
    assert all(row["independent_parse"] for row in result["complete_grammar_controls"])
    assert all(row["independent_exact_audit"]["letters"] > 100 for row in result["complete_grammar_controls"])


def test_center_bridge_alone_is_not_a_sentence_witness():
    grammar = MODULE.CompoundFeatureGrammar()
    row = MODULE.audit(grammar, "Part trap.", "partial_bridge_only", ("part", "trap"))
    assert row["independent_exact_audit"]["exact"]
    assert not row["independent_parse"]
    assert not row["mechanically_admitted"]

