"""Tests that the centre is discovered, rather than installed as a seed."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "independent_leaf_constraint_search_20260913",
    ROOT / "experiments/independent_leaf_constraint_search_20260913.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_center_is_selected_from_independent_typed_channels():
    stats = MODULE.Counter()
    found = MODULE.discover_center(stats)
    assert found is not None
    assert stats["independent_center_assignments"] > 1
    assert stats["center_complete_traces"] == 1
    assert found["words"] == ("to", "order", "red", "root")
    normalized = MODULE.normalize_letters(" ".join(found["words"]))
    assert normalized == normalized[::-1]
    assert all(a[::-1] != b for a in found["words"][:2] for b in found["words"][2:])


def test_discovered_bridge_cancels_every_character_without_whole_word_matching():
    found = MODULE.discover_center(MODULE.Counter())
    bridge = found["trace"]
    assert bridge["completed"]
    assert bridge["cancellations"] == 7
    assert len(bridge["trace"]) == 14
    assert [event["side"] for event in bridge["trace"]] == ["left", "right"] * 7
    assert all(len(event["char"]) == 1 for event in bridge["trace"])
    assert all(event["action"] == "cancel" or not event["residual_before"] for event in bridge["trace"])


def test_fresh_search_reparses_long_controls_and_admits_no_partial_candidate():
    result = MODULE.run(state_limit=100_000, closure_limit=100)
    assert result["config"]["independent_center_leaf_selection"]
    assert not result["config"]["prebuilt_center_palindrome"]
    assert result["discovered_center"]["trace"]["completed"]
    assert result["stats"]["outer_leaf_exposures"] > 0
    assert result["exact_closures"] == []
    assert result["admitted_closures"] == []
    assert all(row["independent_parse"] for row in result["complete_grammar_controls"])
    assert all(row["independent_exact_audit"]["letters"] > 100 for row in result["complete_grammar_controls"])

