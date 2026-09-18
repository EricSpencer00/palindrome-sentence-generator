"""Tests for the non-palindromic center and hard span rejection."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "nonpal_center_crossleaf_residual_20260913",
    ROOT / "experiments/nonpal_center_crossleaf_residual_20260913.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_hard_constraint_rejects_every_contiguous_self_palindromic_span():
    assert MODULE.self_palindromic_multiword_spans(("to", "order", "red", "root"))
    assert MODULE.self_palindromic_multiword_spans(("order", "red")) == []


def test_crossleaf_trace_clears_nonpal_center_by_adjacent_leaf_residual():
    trace = MODULE.crossleaf_trace("order", "red", "to", "root")
    assert trace["completed"]
    assert trace["cancellations"] == 7
    assert len(trace["trace"]) == 14
    assert [event["side"] for event in trace["trace"]] == ["left", "right"] * 7
    assert all(len(event["char"]) == 1 for event in trace["trace"])
    assert trace["trace"][6]["word"] == "order"
    assert trace["trace"][7]["word"] == "root"
    assert trace["trace"][10]["word"] == "to"
    assert trace["trace"][-1]["word"] == "root"


def test_fresh_run_records_rejected_bridge_and_long_reparsed_controls():
    result = MODULE.run(state_limit=100_000, closure_limit=100)
    found = result["discovered_cross_bridge"]
    assert found["center_is_self_palindrome"] is False
    assert found["trace"]["completed"]
    assert found["hard_generator_reject"]
    assert result["config"]["reject_every_self_palindromic_contiguous_multiword_span"]
    assert result["exact_closures"] == []
    assert result["admitted_closures"] == []
    assert all(row["independent_parse"] for row in result["complete_grammar_controls"])
    assert all(row["independent_exact_audit"]["letters"] > 100 for row in result["complete_grammar_controls"])
