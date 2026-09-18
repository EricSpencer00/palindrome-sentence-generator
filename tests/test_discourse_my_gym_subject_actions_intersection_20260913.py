"""Tests for the fresh subject-action ``my ... gym`` construction."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/discourse_my_gym_subject_actions_intersection_20260913.py"
spec = importlib.util.spec_from_file_location("my_gym_subject_actions_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(MODULE)


def test_complete_control_is_intact_parsed_and_semantically_typed():
    grammar = MODULE.SubjectActionGrammar()
    state = MODULE.explicit_control(grammar)
    text = MODULE.BASE.render(state)
    row = MODULE.audit(grammar, text, "test_control", state.trace)
    assert text == (
        "my gardeners tend the healthy plants and water the clean beds and "
        "prune the tall trees and carry the heavy tools to the gym"
    )
    assert row["independent_parse"]
    assert row["feature_witness"]["subject_action_ok"]
    assert row["feature_witness"]["agreement_ok"]
    assert row["feature_witness"]["valency_ok"]
    assert row["independent_exact_audit"]["letters"] >= 100
    assert row["central_admission"]["lexicon_words"]
    assert row["central_admission"]["distinct_words"]
    assert row["central_admission"]["no_self_palindromic_proper_multiword_span"]
    assert row["central_admission"]["not_forbidden_catalogue_endpoint_scaffold"]
    assert row["central_admission"]["absent_from_local_catalogue"]


def test_real_emitter_replays_unequal_endpoint_until_first_contradiction():
    preflight = MODULE.boundary_preflight()
    events = preflight["emitter_events"]
    assert len(events) == 7
    assert [(event["side"], event["character"]) for event in events[:6]] == [
        (1, "m"), (-1, "m"), (1, "y"), (-1, "y"),
        (1, "g"), (-1, "g"),
    ]
    assert events[6]["slot"] == "fixed_gardeners"
    assert events[6]["character"] == "a"
    assert preflight["failed_attempt"]["slot"] == "fixed_the"
    assert preflight["failed_attempt"]["character"] == "e"
    assert preflight["independent_parse"]


def test_bounded_full_scheduler_has_replayable_real_contradiction():
    result = MODULE.run(max_states=2500)
    assert result["config"]["single_shared_derivation_tree"]
    assert result["config"]["endpoint_scaffold_gate"]
    assert result["stats"]["states"] <= 2500
    assert result["states_exhausted"]
    assert result["stats"]["exact_closures"] == 0
    ledger = result["deepest_state_ledger"]
    assert ledger["deepest_contradiction_state"]
    assert ledger["deepest_contradiction_replay"]["same_attempts"]
    assert ledger["deepest_contradiction_state"]["attempts"]
    assert result["complete_subject_action_control"]["provenance"]["shared_tree"]


def test_subject_action_semantics_do_not_accept_a_wrong_patient_frame():
    grammar = MODULE.SubjectActionGrammar()
    state = MODULE.explicit_control(grammar)
    text = MODULE.BASE.render(state).replace("water the clean beds", "water the clean trees")
    row = MODULE.audit(grammar, text, "tampered_control", state.trace)
    assert not row["independent_parse"]
    assert "independent_complete_reparse_failed" in row["rejection_codes"]
