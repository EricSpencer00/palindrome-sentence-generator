"""Focused invariants for the fresh centre-out story grammar lane."""

from __future__ import annotations

from experiments import two_sided_semantic_orbit_product_20260920 as lane


def test_bounded_run_keeps_story_controls_and_independent_audits():
    result = lane.run(max_paths=160, max_states=5_000)

    assert result["target_range"] == [39, 60]
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["exact"] == 0
    assert result["stats"]["intact_controls"] == 4
    assert result["rendered_candidates"] == []
    assert result["rendered_controls"]
    for row in result["all_rendered_rows"]:
        audit = row["audit"]
        assert audit["two_pointer_exact"] is False
        assert audit["sha256_equal"] is False
        assert audit["sha256_forward"] != audit["sha256_reverse"]
        assert row["complete_paths"]["left"]["complete_finite_semantic_path"]
        assert row["complete_paths"]["right"]["complete_finite_semantic_path"]
        assert row["provenance"]["grammar_boundaries_selected_before_render"]
        assert row["provenance"]["semantic_roles_selected_before_render"]
        assert row["orbit_assignment"]["first_failure"]


def test_held_out_setting_frame_is_complete_and_ordinary_order():
    paths = lane.build_paths("right", max_paths=200, max_letters=40)
    setting_paths = [path for path in paths if path.frame == "setting_svo"]
    assert setting_paths
    assert all(path.complete_finite_semantics for path in setting_paths)
    assert any("near the " in path.text or "at the " in path.text for path in setting_paths)
    assert any(
        marker in f" {path.text} "
        for marker in (" at noon ", " at winter ")
        for path in setting_paths
    )
    assert all(
        segment.role in {"SETTING_PREP", "SETTING_DET", "SETTING_OBJECT", "SUBJECT_DET", "SUBJECT", "FINITE_VERB", "OBJECT_DET", "OBJECT"}
        for path in setting_paths[:20]
        for segment in path.segments
    )


def test_product_starts_at_the_clause_centre_and_supports_odd_length():
    # The repeated letter is only a compact orientation fixture; the
    # production bank is still role-complete and lexicon-gated.  ``left`` has
    # one unpaired centre character, then 19 matched orbits give a 39-letter
    # palindrome from the left end/right beginning cursor pair.
    left = lane.ScenePath(
        "left-fixture",
        "left",
        "svo",
        (lane.Segment("SUBJECT", "a" * 20),),
    )
    right = lane.ScenePath(
        "right-fixture",
        "right",
        "svo",
        (lane.Segment("OBJECT", "a" * 19),),
    )
    product = lane._orbit_product(
        lane.PathTrie((left,), reverse_cursor=True),
        lane.PathTrie((right,), reverse_cursor=False),
        min_letters=39,
        max_letters=39,
        max_states=200,
    )

    assert product["budget_exhausted"] is False
    # The product also explores the even and right-centre branches; the
    # accepted row below proves that the left-centre branch reached all 19
    # paired orbits.
    assert product["matched_orbit_transitions"] >= 19
    assert product["closure_pairs"] == [{
        "left": 0,
        "right": 0,
        "center_mode": "left_center",
        "paired_orbits": 19,
        "letters": 39,
    }]
