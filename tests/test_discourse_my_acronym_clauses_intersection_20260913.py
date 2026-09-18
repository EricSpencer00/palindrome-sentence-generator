"""Tests for the explicit my/acronym discourse route."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("discourse_my_acronym_clauses_intersection_20260913", ROOT / "experiments/discourse_my_acronym_clauses_intersection_20260913.py")
MODULE = importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE; SPEC.loader.exec_module(MODULE)


def test_real_endpoint_replay_crosses_my_into_longer_acronym():
    row = MODULE.boundary_preflight(); events = row["emitter_events"]
    assert row["independent_parse"] and row["failed_attempt"]["emitter_rejected"]
    assert len(events) == 9
    assert [(events[i]["character"], events[i + 1]["character"]) for i in range(0, 8, 2)] == [("m", "m"), ("y", "y"), ("n", "n"), ("o", "o")]
    assert events[1]["word"] == "acronym" and row["failed_attempt"]["word"] == "acronym"


def test_control_has_explicit_note_taking_referents_and_hard_gates():
    grammar = MODULE.MyAcronymGrammar(); state = MODULE.explicit_control(grammar); text = MODULE.BASE.render(state); row = MODULE.audit(grammar, text, "complete_explicit_discourse_control", state.trace)
    assert text.startswith("my notes describe the clean map") and text.endswith("form the final acronym")
    assert row["independent_parse"] and row["feature_witness"]["valency_ok"] and 100 <= row["independent_exact_audit"]["letters"] <= 180
    assert row["central_admission"]["lexicon_words"] and row["central_admission"]["distinct_words"] and row["central_admission"]["no_self_palindromic_proper_multiword_span"]


def test_full_scheduler_records_replayable_contradiction():
    result = MODULE.solver(MODULE.MyAcronymGrammar(), max_states=2500); ledger = result["deepest_state_ledger"]; state = ledger["deepest_contradiction_state"]
    assert state and ledger["deepest_contradiction_replay"]["same_attempts"]
    assert state["rendered_tree"].startswith("my notes") and any(a.get("available") and not a.get("accepted") for a in state["attempts"])
