"""Tests for the fresh connected-tree lexical zipper."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/zipper_typed_clause_intersection_20260913.py"
spec = importlib.util.spec_from_file_location("typed_zipper_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = MODULE
spec.loader.exec_module(MODULE)


def test_control_is_short_intact_ordinary_and_independently_typed():
    grammar = MODULE.ZipperGrammar()
    state = MODULE.explicit_control(grammar)
    text = MODULE.render(state)
    row = MODULE.audit(grammar, text, "test_control", state.trace)
    assert text == "a red mechanic repairs a blue camera"
    assert row["independent_parse"]
    assert row["feature_witness"] == {
        "subject_role": "person", "subject_number": "sing",
        "verb_frame": "transitive", "object_role": "object",
        "event": "repair", "agreement_ok": True,
        "valency_ok": True, "subject_action_ok": True,
        "complete_tree": True,
    }
    assert row["independent_exact_audit"]["letters"] == 30
    assert row["central_admission"]["lexicon_words"]
    assert row["central_admission"]["distinct_words"]
    assert row["central_admission"]["no_self_palindromic_proper_multiword_span"]
    assert row["central_admission"]["not_forbidden_catalogue_endpoint_scaffold"]
    assert row["central_admission"]["absent_from_local_catalogue"]


def test_zipper_replays_real_live_residual_and_rejects_at_next_pair():
    preflight = MODULE.zipper_preflight(MODULE.ZipperGrammar())
    events = preflight["emitter_events"]
    assert [(event["side"], event["character"]) for event in events[:6]] == [
        (1, "a"), (-1, "a"), (1, "r"), (-1, "r"),
        (1, "e"), (-1, "e"),
    ]
    assert events[-1]["residual_after"] == "d"
    assert preflight["failed_attempt"] == {
        "step": 8, "side": -1, "slot": "patient", "word": "camera",
        "character": "m", "emitter_rejected": True,
    }
    assert preflight["independent_parse"]


def test_full_scheduler_uses_live_residual_filtered_lexical_choices():
    result = MODULE.run(max_states=100000)
    assert result["config"]["bidirectional_zipper"]
    assert result["config"]["lexical_choice_requires_live_residual"]
    assert result["config"]["endpoint_scaffold_gate"]
    assert result["stats"]["states"] < 100000
    assert result["states_exhausted"]
    assert result["stats"]["exact_closures"] == 0
    ledger = result["deepest_state_ledger"]["deepest_contradiction_state"]
    assert ledger["zipper_choices"]
    assert any(choice[0] == -1 and choice[3] == "t" for choice in ledger["zipper_choices"])


def test_wrong_typed_patient_is_rejected_by_independent_reparse():
    grammar = MODULE.ZipperGrammar()
    state = MODULE.explicit_control(grammar)
    wrong = MODULE.render(state).replace("a blue camera", "a blue table")
    row = MODULE.audit(grammar, wrong, "tampered_control", state.trace)
    assert not row["independent_parse"]
    assert "independent_complete_reparse_failed" in row["rejection_codes"]
