"""Tests for typed boundary-role relexicalization."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/typed_boundary_relexicalization_reason_transducer_20260913.py"
spec = importlib.util.spec_from_file_location("boundary_relex_transducer_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = MODULE
spec.loader.exec_module(MODULE)


def test_compatibility_automaton_enumerates_roles_before_traversal():
    rows = MODULE.outer_role_compatibilities()
    assert len(rows) == len(MODULE.FRAMES) * 2
    tablet = [row for row in rows if row["event"] == "tablet_data" and row["left_word"] == "a"][0]
    assert tablet["right_word"] == "data"
    assert tablet["compatible_prefix"] == "a"
    assert tablet["left_word_complete"]
    assert not tablet["right_word_complete"]
    assert all("left_role" in row and "right_role" in row and "prefix_letters" in row for row in rows)


def test_grammar_generated_prose_reparses_and_has_semantic_witness():
    for frame in MODULE.FRAMES:
        text = MODULE.generated_surface(frame)
        row = MODULE.audit(text, "grammar_generated_diagnostic")
        assert row["independent_parse"]
        assert row["feature_witness"]["subject_action_ok"]
        assert row["feature_witness"]["agreement_ok"]
        assert row["feature_witness"]["valency_ok"]
        assert row["independent_exact_audit"]["letters"] >= 30
        assert all(value for key, value in row["central_admission"].items()
                   if key != "exact_letter_palindrome")
        assert row["central_admission"]["not_forbidden_catalogue_endpoint_scaffold"]


def test_partial_word_transducer_crosses_a_role_boundary_and_exhausts():
    result = MODULE.run(max_states=100000)
    assert result["config"]["boundary_role_compatibility_automaton"]
    assert result["config"]["forward_and_reverse_lexical_tries"]
    assert result["config"]["advance_role_only_on_word_completion"]
    assert result["config"]["preauthored_full_control"] is False
    assert result["states_exhausted"]
    assert result["rendered_candidates"] == []
    assert result["stats"]["exact_closures"] == 0
    deepest = result["deepest_state_ledger"]
    assert deepest["event"] == "tablet_data"
    assert deepest["length"] == 3
    assert deepest["pair_trace"] == [[1, "a"], [2, "t"], [3, "a"]]
    assert "left_complete:subject_det:a" in deepest["role_trace"]
    assert deepest["next_pair_count"] == 0
    assert result["deepest_contradiction"]["rejection"]


def test_incompatible_event_surface_fails_independent_parse():
    wrong = MODULE.generated_surface(MODULE.FRAMES[0]).replace("flashes", "rings")
    row = MODULE.audit(wrong, "tampered_event")
    assert not row["independent_parse"]
    assert "independent_complete_reparse_failed" in row["rejection_codes"]
    assert "subject_action_semantics_failure" in row["rejection_codes"]
