"""Tests for the two-sentence shifted-boundary repair run."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "discourse_shifted_boundary_local_repair_20260913",
    ROOT / "experiments/discourse_shifted_boundary_local_repair_20260913.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_detector_runs_before_forbidden_repair_is_ledger_eligible():
    assert MODULE.forbidden_spans(("to", "iron", "no", "riot"))
    found = MODULE.discover_repair(MODULE.Counter())
    assert found["status"] == "repair_blocked_by_hard_span"
    assert found["repair"]["blocked_before_ledger"]
    assert found["repair"]["trace"] is None


def test_shifted_discourse_boundary_clears_residual_live_across_leaves():
    trace = MODULE.live_seam("iron", "no", "riot", "to")
    assert trace["completed"]
    assert trace["cancellations"] == 6
    assert len(trace["ledger"]) == 12
    assert [e["side"] for e in trace["ledger"]] == ["left", "right"] * 6
    assert all(len(e["char"]) == 1 for e in trace["ledger"])
    assert trace["ledger"][0]["word"] == "iron"
    assert trace["ledger"][2]["word"] == "iron"
    assert trace["ledger"][8]["word"] == "to"
    assert trace["replay"]["ok"]


def test_failed_frontier_is_concrete_and_controls_reparse():
    result = MODULE.run(state_limit=100_000, closure_limit=100)
    initial = result["discovery"]["initial"]["trace"]
    assert initial["first_incompatible"]["expected"] == "t"
    assert initial["first_incompatible"]["char"] == "n"
    assert initial["replay"]["events_replayed"] == 2
    assert result["exact_closures"] == []
    assert result["admitted_closures"] == []
    assert all(row["independent_parse"] for row in result["complete_grammar_controls"])
    assert all(row["independent_exact_audit"]["letters"] > 100 for row in result["complete_grammar_controls"])
    assert all(row["rendered"].count(".") == 2 for row in result["complete_grammar_controls"])
