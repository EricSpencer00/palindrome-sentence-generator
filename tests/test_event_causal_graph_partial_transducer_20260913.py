"""Tests for graph-checked partial-word transduction."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/event_causal_graph_partial_transducer_20260913.py"
spec = importlib.util.spec_from_file_location("event_graph_transducer_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = MODULE
spec.loader.exec_module(MODULE)


def test_graph_replay_checks_agent_action_target_and_effect():
    graph = MODULE.GRAPHS[0]
    words = tuple(slot.words[0] for slot in MODULE.graph_slots(graph))
    replay = MODULE.graph_replay(graph, words)
    assert replay["replay_ok"]
    assert replay["cause_event"]["agent"] == "sensor"
    assert replay["cause_event"]["action"] == "detects"
    assert replay["cause_event"]["target"] == "spark"
    assert replay["effect_event"] == {"agent": "monitor", "action": "records", "target": "signal"}
    assert replay["mechanism"] == "detected_event"
    assert not MODULE.graph_replay(graph, words[:-1] + ("record",))["replay_ok"]


def test_graph_generated_surfaces_parse_and_pass_all_nonexact_gates():
    for graph in MODULE.GRAPHS:
        text = MODULE.generated_surface(graph)
        row = MODULE.audit(text, "graph_generated_diagnostic")
        assert row["independent_parse"]
        assert row["feature_witness"]["causal_graph_replay_ok"]
        assert row["feature_witness"]["subject_action_ok"]
        assert row["central_admission"]["lexicon_words"]
        assert row["central_admission"]["distinct_words"]
        assert row["central_admission"]["not_forbidden_catalogue_endpoint_scaffold"]
        assert row["central_admission"]["absent_from_local_catalogue"]
        assert row["independent_exact_audit"]["letters"] >= 30


def test_graph_transducer_reaches_multiple_word_boundary_state_then_exhausts():
    result = MODULE.run(max_states=100000)
    assert result["config"]["event_causal_graph_required"]
    assert result["config"]["independent_graph_replay_before_traversal"]
    assert result["config"]["partial_word_transducer"]
    assert result["config"]["advance_role_only_on_word_completion"]
    assert result["states_exhausted"]
    assert result["stats"]["exact_closures"] == 0
    assert result["rendered_candidates"] == []
    deepest = result["deepest_state_ledger"]
    assert deepest["graph"] == "tablet_note_reader"
    assert deepest["length"] == 3
    assert deepest["pair_trace"] == [[1, "a"], [2, "t"], [3, "a"]]
    assert "left_complete:cause_det:a" in deepest["role_trace"]
    assert result["deepest_contradiction"]["rejection"]


def test_incompatible_surface_fails_independent_graph_parse():
    graph = MODULE.GRAPHS[0]
    wrong = MODULE.generated_surface(graph).replace("detects", "stores")
    row = MODULE.audit(wrong, "tampered_graph_surface")
    assert not row["independent_parse"]
    assert "independent_complete_reparse_failed" in row["rejection_codes"]
    assert "causal_graph_replay_failure" in row["rejection_codes"]

