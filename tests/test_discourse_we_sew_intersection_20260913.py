"""Tests for the distinct lexicon-backed we/sew route."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("discourse_we_sew_intersection_20260913", ROOT / "experiments/discourse_we_sew_intersection_20260913.py")
MODULE = importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE; SPEC.loader.exec_module(MODULE)


def test_real_emitter_crosses_we_into_sew_and_start():
    row = MODULE.boundary_preflight(); events = row["emitter_events"]
    assert row["independent_parse"] and row["failed_attempt"]["emitter_rejected"]
    assert len(events) == 7
    assert [(events[i]["character"], events[i + 1]["character"]) for i in range(0, 6, 2)] == [("w", "w"), ("e", "e"), ("s", "s")]
    assert events[1]["word"] == "sew" and events[5]["word"] == "sew"
    assert row["failed_attempt"]["word"] == "and"


def test_long_control_is_lexicon_backed_and_reparsed():
    grammar = MODULE.WeSewGrammar(5); state = MODULE.explicit_control(grammar); text = MODULE.BASE.render(state); row = MODULE.audit(grammar, text, "complete_we_sew_recursive_control", state.trace)
    assert row["independent_parse"] and row["feature_witness"]["agreement_ok"] and row["feature_witness"]["valency_ok"]
    assert 100 <= row["independent_exact_audit"]["letters"] <= 180 and row["central_admission"]["lexicon_words"]
    assert row["central_admission"]["no_self_palindromic_proper_multiword_span"]


def test_full_scheduler_has_replayable_contradiction():
    result = MODULE.solver(MODULE.WeSewGrammar(6), max_states=2500); ledger = result["deepest_state_ledger"]; state = ledger["deepest_contradiction_state"]
    assert state and ledger["deepest_contradiction_replay"]["same_attempts"]
    assert state["rendered_tree"].startswith("we start") and any(a.get("available") and not a.get("accepted") for a in state["attempts"])
