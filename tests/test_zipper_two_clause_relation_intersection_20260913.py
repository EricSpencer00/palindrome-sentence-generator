"""Tests for the connected two-clause zipper relation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/zipper_two_clause_relation_intersection_20260913.py"
spec = importlib.util.spec_from_file_location("two_clause_zipper_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = MODULE
spec.loader.exec_module(MODULE)


def test_two_clause_control_is_intact_and_independently_typed():
    grammar = MODULE.TwoClauseGrammar()
    state = MODULE.explicit_control(grammar)
    text = MODULE.render(state)
    row = MODULE.audit(grammar, text, "test_control", state.trace)
    assert text == "we record a brief survey while the skilled clerks use a plain screw"
    assert row["independent_parse"]
    assert row["feature_witness"] == {
        "relation": "record_use", "agreement_ok": True,
        "valency_ok": True, "subject_action_ok": True,
        "left_subject_type": "person", "right_subject_type": "person",
        "left_object_type": "document", "right_object_type": "implement",
        "complete_tree": True,
    }
    assert 30 <= row["independent_exact_audit"]["letters"] <= 60
    assert row["central_admission"]["lexicon_words"]
    assert row["central_admission"]["distinct_words"]
    assert row["central_admission"]["no_self_palindromic_proper_multiword_span"]
    assert row["central_admission"]["not_forbidden_catalogue_endpoint_scaffold"]
    assert row["central_admission"]["absent_from_local_catalogue"]


def test_real_zipper_crosses_outer_words_before_rejecting():
    preflight = MODULE.boundary_preflight(MODULE.TwoClauseGrammar())
    events = preflight["emitter_events"]
    assert [(event["side"], event["character"]) for event in events[:6]] == [
        (1, "w"), (-1, "w"), (1, "e"), (-1, "e"),
        (1, "r"), (-1, "r"),
    ]
    assert events[-1]["residual_after"] == "e"
    assert preflight["failed_attempt"] == {
        "step": 8, "side": -1, "slot": "right_object", "word": "screw",
        "character": "c", "emitter_rejected": True,
    }
    assert preflight["independent_parse"]


def test_full_relation_search_records_joint_connector_and_live_choice():
    result = MODULE.run(max_states=100000)
    assert result["config"]["single_shared_derivation_tree"]
    assert result["config"]["central_relation_jointly_lexicalized"]
    assert result["config"]["outer_constituents_jointly_lexicalized"]
    assert result["config"]["lexical_choice_requires_live_residual"]
    assert result["config"]["endpoint_scaffold_gate"]
    assert result["states_exhausted"]
    assert result["stats"]["exact_closures"] == 0
    ledger = result["deepest_state_ledger"]["deepest_contradiction_state"]
    assert ledger["zipper_choices"]
    assert any(choice[0] == -1 and choice[3] == "w"
               for choice in ledger["zipper_choices"])
    assert any(choice[1] == "CONNECTOR" for choice in ledger["zipper_choices"])


def test_wrong_clause_frame_fails_independent_reparse():
    grammar = MODULE.TwoClauseGrammar()
    state = MODULE.explicit_control(grammar)
    wrong = MODULE.render(state).replace("use a plain screw", "use a plain survey")
    row = MODULE.audit(grammar, wrong, "tampered_control", state.trace)
    assert not row["independent_parse"]
    assert "independent_complete_reparse_failed" in row["rejection_codes"]

