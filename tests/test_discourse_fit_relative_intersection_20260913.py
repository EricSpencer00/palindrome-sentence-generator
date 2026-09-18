"""Tests for the distinct two-sentence discourse CFG route."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "discourse_fit_relative_intersection_20260913",
    ROOT / "experiments/discourse_fit_relative_intersection_20260913.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_joint_discourse_envelope_replays_real_opening_contact():
    row = MODULE.boundary_preflight(MODULE.DiscourseGrammar(6))
    assert row["independent_parse"] and not row["exact_audit"]["exact"]
    events = row["emitter_events"]
    assert len(events) == 9
    assert [(events[i]["character"], events[i + 1]["character"]) for i in range(0, 8, 2)] == [
        ("d", "d"), ("i", "i"), ("d", "d"), ("i", "i")
    ]
    assert row["failed_attempt"]["emitter_rejected"]
    assert (row["failed_attempt"]["word"], row["failed_attempt"]["character"]) == ("watched", "d")


def test_discourse_probe_is_a_complete_typed_reparse():
    grammar = MODULE.DiscourseGrammar(1)
    text = "did i examine this quiet portrait that a quiet teacher watched? i did."
    tree = MODULE.parse_discourse(grammar, text)
    witness = MODULE.semantic_witness(tree)
    assert tree is not None and witness["complete_tree"]
    assert witness["agreement_ok"] and witness["valency_ok"]
    assert witness["relative_count"] == 1


def test_long_control_has_independent_reparse_and_hard_span_gate():
    grammar = MODULE.DiscourseGrammar(6); result = MODULE.run(max_states=1000)
    control = result["complete_recursive_control"]
    assert control["independent_parse"] and control["feature_witness"]["agreement_ok"]
    assert control["feature_witness"]["valency_ok"]
    assert control["independent_exact_audit"]["letters"] >= 100
    assert control["central_admission"]["no_self_palindromic_proper_multiword_span"]


def test_deepest_rejection_is_a_replayed_live_emission():
    result = MODULE.solver(MODULE.DiscourseGrammar(6), max_states=5000)
    ledger = result["deepest_state_ledger"]
    state = ledger["deepest_contradiction_state"]
    replay = ledger["deepest_contradiction_replay"]
    assert state is not None and replay is not None
    assert replay["same_rendered_tree"] and replay["same_length"]
    assert replay["same_residual"] and replay["same_attempts"]
    rejected = [attempt for attempt in state["attempts"] if attempt.get("available") and not attempt.get("accepted")]
    assert rejected
    # This is the next live character boundary after the authored ``statue``
    # morphology repair, not a scripted seam.
    assert rejected[0]["slot"] == "fixed_examine"
    assert rejected[0]["character"] == "x"
    assert state["residual"] == "u"
