"""Tests for the crossed-boundary connected-tree discourse route."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("discourse_now_unwon_intersection_20260913", ROOT / "experiments/discourse_now_unwon_intersection_20260913.py")
MODULE = importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE; SPEC.loader.exec_module(MODULE)


def test_outer_contact_crosses_longer_word_through_real_emitter():
    row = MODULE.boundary_preflight(); events = row["emitter_events"]
    assert row["independent_parse"] and row["failed_attempt"]["emitter_rejected"]
    assert len(events) == 11
    assert [(events[i]["character"], events[i + 1]["character"]) for i in range(0, 10, 2)] == [("n", "n"), ("o", "o"), ("w", "w"), ("n", "n"), ("u", "u")]
    assert events[1]["word"] == "unwon" and events[9]["word"] == "unwon"
    assert row["failed_attempt"]["word"] == "remains"


def test_long_control_is_intact_typed_prose_and_independently_reparsed():
    result = MODULE.run(max_states=1000); control = result["complete_recursive_control"]
    assert control["independent_parse"] and control["feature_witness"]["agreement_ok"] and control["feature_witness"]["valency_ok"]
    assert 100 <= control["independent_exact_audit"]["letters"] <= 180
    assert control["central_admission"]["no_self_palindromic_proper_multiword_span"]
    assert control["central_admission"]["distinct_words"]


def test_bounded_solver_records_replayable_full_tree_contradiction():
    result = MODULE.solver(MODULE.CrossBoundaryGrammar(5), max_states=3000)
    ledger = result["deepest_state_ledger"]; state = ledger["deepest_contradiction_state"]
    assert state and ledger["deepest_contradiction_replay"]["same_attempts"]
    assert state["rendered_tree"].startswith("now nurses record")
    assert any(a.get("available") and not a.get("accepted") for a in state["attempts"])
