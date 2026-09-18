"""Tests for the explicit can/cognac question route."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("question_can_cognac_clauses_intersection_20260913", ROOT / "experiments/question_can_cognac_clauses_intersection_20260913.py")
MODULE = importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE; SPEC.loader.exec_module(MODULE)


def test_real_endpoint_replay_crosses_can_into_longer_cognac():
    row = MODULE.boundary_preflight(); events = row["emitter_events"]
    assert row["independent_parse"] and row["failed_attempt"]["emitter_rejected"]
    assert len(events) == 11
    assert [(events[i]["character"], events[i + 1]["character"]) for i in range(0, 10, 2)] == [("c", "c"), ("a", "a"), ("n", "n"), ("g", "g"), ("o", "o")]
    assert events[1]["word"] == "cognac" and row["failed_attempt"]["word"] == "cognac"


def test_control_has_explicit_connected_action_referents():
    grammar = MODULE.CanCognacGrammar(); state = MODULE.explicit_control(grammar); text = MODULE.BASE.render(state); row = MODULE.audit(grammar, text, "complete_explicit_question_control", state.trace)
    assert text.startswith("can golfers carry the clean basket") and text.endswith("serve the strong cognac")
    assert row["independent_parse"] and row["feature_witness"]["valency_ok"] and 100 <= row["independent_exact_audit"]["letters"] <= 180
    assert row["central_admission"]["lexicon_words"] and row["central_admission"]["distinct_words"] and row["central_admission"]["no_self_palindromic_proper_multiword_span"]


def test_full_scheduler_records_replayable_contradiction():
    result = MODULE.solver(MODULE.CanCognacGrammar(), max_states=2500); ledger = result["deepest_state_ledger"]; state = ledger["deepest_contradiction_state"]
    assert state and ledger["deepest_contradiction_replay"]["same_attempts"]
    assert state["rendered_tree"].startswith("can golfers") and any(a.get("available") and not a.get("accepted") for a in state["attempts"])
