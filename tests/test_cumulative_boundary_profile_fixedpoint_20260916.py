import re

from experiments.cumulative_boundary_profile_fixedpoint_20260916 import (
    independent_audit,
    joint_emit,
    profile_class_expansion_templates,
    run,
)


def test_joint_state_tracks_symmetric_boundaries_and_cross_boundary_matches():
    words = ("the", "bear", "saw", "the", "book", "near", "the", "zoo",
             "and", "the", "wolf", "held", "the", "map", "near", "sea")
    state = joint_emit(words)
    assert state.lengths == state.lengths[::-1]
    assert state.boundary_pairs >= 2
    assert state.mismatches > 0


def test_profile_run_has_real_long_rows_and_independent_exactness_audit():
    result = run(per_template=8)
    assert result["novelty_preflight"]["status"] == "registered_self"
    assert result["stats"]["rendered_candidates"] > 8
    assert result["stats"]["over_length_candidates"] == result["stats"]["rendered_candidates"]
    assert result["stats"]["exact_count"] == 0
    for row in result["rendered_rows"]:
        assert row["letters"] > 38
        assert row["boundary_profile"]["symmetric"]
        assert row["boundary_profile"]["word_length_sequence_palindrome"]
        assert row["joint_character_ledger"]["cross_boundary_matched_pairs"] >= 2
        assert row["independent_audit"] == independent_audit(row["rendered"])
        assert row["no_self_palindromic_proper_multiword_span"]
        assert re.fullmatch(r"[A-Za-z][A-Za-z ,.]+", row["rendered"])
    assert result["next_operator"]["status"] == "implemented_and_pending"


def test_next_operator_is_a_real_profile_class_expansion():
    expanded = profile_class_expansion_templates()
    assert expanded[0].name.endswith("profile_class_expanded")
    assert "horse" in expanded[0].slots[1].words
    assert "harbor" in expanded[0].slots[7].words
