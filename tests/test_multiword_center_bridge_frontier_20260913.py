"""Invariants for the live multiword centre bridge experiment."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "multiword_center_bridge_frontier_20260913",
    ROOT / "experiments/multiword_center_bridge_frontier_20260913.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_center_phrase_is_normal_multileaf_bridge_without_reverse_word_pair():
    phrase = MODULE.normalize_words(MODULE.BRIDGE_LEFT + MODULE.BRIDGE_RIGHT)
    assert phrase == "toorderredroot"
    assert phrase == phrase[::-1]
    assert all(a[::-1] != b for a in MODULE.BRIDGE_LEFT for b in MODULE.BRIDGE_RIGHT)
    assert MODULE.BRIDGE_LICENSE[MODULE.BRIDGE_LEFT + MODULE.BRIDGE_RIGHT]


def test_live_trace_clears_all_four_center_leaves_one_character_at_a_time():
    bridge = MODULE.bridge_trace()
    events = bridge["trace"]
    assert bridge["completed"]
    assert len(events) == 14
    assert bridge["cancellations"] == 7
    assert [event["side"] for event in events] == ["left", "right"] * 7
    assert all(len(event["char"]) == 1 for event in events)
    assert all(event["action"] == "cancel" or not event["residual_before"] for event in events)
    assert {event["word"] for event in events} == {"order", "to", "red", "root"}


def test_bridge_is_at_actual_midpoint_of_connected_tree():
    grammar = MODULE.Grammar()
    leaves = MODULE.slots(grammar.expand(grammar.start()))
    centre = len(leaves) // 2
    assert len(leaves) == 32
    assert [slot.symbol.role for slot in leaves[centre - 2:centre + 2]] == [
        "bridge_inf", "bridge_verb", "bridge_adj", "bridge_noun"
    ]


def test_fresh_100k_search_exposes_outer_frontier_after_bridge_and_parses_controls():
    result = MODULE.run(state_limit=100_000, closure_limit=100)
    assert result["config"]["center_bridge_before_outer_frontier"]
    assert result["center_bridge_traces"][0]["completed"]
    assert result["stats"]["center_cancellations"] == 7
    assert result["stats"]["outer_leaf_exposures"] > 0
    assert result["exact_closures"] == []
    assert result["stats"]["admitted_closures"] == 0
    assert all(row["independent_parse"] for row in result["complete_grammar_controls"])
    assert all(row["independent_exact_audit"]["letters"] > 100 for row in result["complete_grammar_controls"])

