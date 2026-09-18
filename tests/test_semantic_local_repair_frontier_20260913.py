"""Tests for semantic local repair and replayed character ledgers."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "semantic_local_repair_frontier_20260913",
    ROOT / "experiments/semantic_local_repair_frontier_20260913.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_hard_span_filter_rejects_all_multiword_self_palindromes():
    bad = MODULE.spans(("to", "order", "red", "root"))
    assert bad and bad[0]["start"] == 0 and bad[0]["end"] == 4
    assert MODULE.spans(("order", "red")) == []


def test_local_repair_replays_ledger_and_clears_frontier():
    stats = MODULE.Counter()
    found = MODULE.discover_with_local_repair(stats)
    assert found["status"] == "repaired_but_hard_span_rejected"
    assert found["initial"]["trace"]["completed"] is False
    assert found["initial"]["trace"]["first_incompatible"]["expected"] == "t"
    repair = found["repair"]
    assert repair["operator"] == "semantic_center_verb_substitution"
    assert repair["trace"]["completed"]
    assert repair["ledger_replay_match"]
    assert repair["hard_span_reject"]
    assert repair["trace"]["cancellations"] == 7
    assert all(len(event["char"]) == 1 for event in repair["trace"]["ledger"])


def test_fresh_100k_run_has_only_reparsed_diagnostic_controls():
    result = MODULE.run(state_limit=100_000, closure_limit=100)
    assert result["config"]["semantic_local_repair"]
    assert result["config"]["replayed_ledger"]
    assert result["config"]["reject_every_self_palindromic_contiguous_multiword_span"]
    assert result["exact_closures"] == []
    assert result["admitted_closures"] == []
    assert all(row["independent_parse"] for row in result["complete_grammar_controls"])
    assert all(row["independent_exact_audit"]["letters"] > 100 for row in result["complete_grammar_controls"])

