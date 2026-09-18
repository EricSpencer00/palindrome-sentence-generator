"""Tests for the anaphoric clarification zipper."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/zipper_anaphoric_clarification_intersection_20260913.py"
spec = importlib.util.spec_from_file_location("anaphoric_zipper_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = MODULE
spec.loader.exec_module(MODULE)


def test_anaphoric_control_is_ordinary_and_independently_reparsed():
    grammar = MODULE.AnaphoricGrammar()
    state = MODULE.explicit_control(grammar)
    text = MODULE.render(state)
    row = MODULE.audit(grammar, text, "test_control", state.trace)
    assert text == "an expert found a useful map and it explains the arena"
    assert row["independent_parse"]
    assert row["feature_witness"] == {
        "discourse_function": "anaphoric_clarification", "antecedent": "map",
        "agreement_ok": True, "valency_ok": True, "subject_action_ok": True,
        "coreference_ok": True, "complete_tree": True,
    }
    assert 30 <= row["independent_exact_audit"]["letters"] <= 60
    assert row["central_admission"]["lexicon_words"]
    assert row["central_admission"]["distinct_words"]
    assert row["central_admission"]["no_self_palindromic_proper_multiword_span"]
    assert row["central_admission"]["not_forbidden_catalogue_endpoint_scaffold"]
    assert row["central_admission"]["absent_from_local_catalogue"]


def test_real_zipper_records_three_crossed_characters_then_rejects():
    preflight = MODULE.boundary_preflight(MODULE.AnaphoricGrammar())
    events = preflight["emitter_events"]
    assert [(event["side"], event["character"]) for event in events[:6]] == [
        (1, "a"), (-1, "a"), (1, "n"), (-1, "n"),
        (1, "e"), (-1, "e"),
    ]
    assert events[-1]["residual_after"] == "x"
    assert preflight["failed_attempt"] == {
        "step": 8, "side": -1, "slot": "patient", "word": "arena",
        "character": "r", "emitter_rejected": True,
    }
    assert preflight["independent_parse"]


def test_full_search_keeps_anaphoric_relation_and_live_lexical_choice():
    result = MODULE.run(max_states=100000)
    assert result["config"]["single_shared_derivation_tree"]
    assert result["config"]["central_discourse_seam_jointly_lexicalized"]
    assert result["config"]["outer_constituents_jointly_lexicalized"]
    assert result["config"]["anaphoric_coreference_required"]
    assert result["config"]["lexical_choice_requires_live_residual"]
    assert result["config"]["endpoint_scaffold_gate"]
    assert result["states_exhausted"]
    assert result["stats"]["exact_closures"] == 0
    ledger = result["deepest_state_ledger"]["deepest_contradiction_state"]
    assert ledger["zipper_choices"]
    assert any(choice[0] == -1 and choice[3] == "a" for choice in ledger["zipper_choices"])
    assert result["complete_anaphoric_clarification_control"]["feature_witness"]["coreference_ok"]


def test_wrong_anaphoric_object_is_rejected_by_independent_parse():
    grammar = MODULE.AnaphoricGrammar()
    state = MODULE.explicit_control(grammar)
    wrong = MODULE.render(state).replace("the arena", "the terrain")
    row = MODULE.audit(grammar, wrong, "tampered_control", state.trace)
    assert not row["independent_parse"]
    assert "independent_complete_reparse_failed" in row["rejection_codes"]
    assert "coreference_failure" in row["rejection_codes"]
