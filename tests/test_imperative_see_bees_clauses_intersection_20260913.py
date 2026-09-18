"""Tests for the explicit see/bees imperative route."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("imperative_see_bees_clauses_intersection_20260913", ROOT / "experiments/imperative_see_bees_clauses_intersection_20260913.py")
MODULE = importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE; SPEC.loader.exec_module(MODULE)


def test_real_endpoint_replay_crosses_see_into_longer_bees():
    row = MODULE.boundary_preflight(); events = row["emitter_events"]
    assert row["independent_parse"] and row["failed_attempt"]["emitter_rejected"]
    assert len(events) == 7
    assert [(events[i]["character"], events[i + 1]["character"]) for i in range(0, 6, 2)] == [("s", "s"), ("e", "e"), ("e", "e")]
    assert events[1]["word"] == "bees" and row["failed_attempt"]["word"] == "bees"


def test_control_has_explicit_connected_referents_and_hard_gates():
    grammar = MODULE.SeeBeesGrammar(); state = MODULE.explicit_control(grammar); text = MODULE.BASE.render(state); row = MODULE.audit(grammar, text, "complete_explicit_imperative_control", state.trace)
    assert text.startswith("see the keeper carry the clean basket") and text.endswith("feed the hungry bees")
    assert row["independent_parse"] and row["feature_witness"]["valency_ok"] and 100 <= row["independent_exact_audit"]["letters"] <= 180
    assert row["central_admission"]["lexicon_words"] and row["central_admission"]["distinct_words"] and row["central_admission"]["no_self_palindromic_proper_multiword_span"]


def test_full_scheduler_records_replayable_contradiction():
    result = MODULE.solver(MODULE.SeeBeesGrammar(), max_states=2500); ledger = result["deepest_state_ledger"]; state = ledger["deepest_contradiction_state"]
    assert state and ledger["deepest_contradiction_replay"]["same_attempts"]
    assert state["rendered_tree"].startswith("see the keeper") and any(a.get("available") and not a.get("accepted") for a in state["attempts"])
