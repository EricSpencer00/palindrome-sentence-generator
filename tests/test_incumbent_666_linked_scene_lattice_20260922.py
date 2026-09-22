import json

from experiments.incumbent_666_linked_scene_lattice_20260922 import (
    OUT,
    SCENE_LEFT,
    SCENE_RIGHT,
    normalize,
)


def test_bounded_linked_scene_closes_and_preserves_72_letter_seam_equation():
    payload = json.loads(OUT.read_text())
    row = payload["rows"][0]
    attempt = row["linked_scene_attempt"]
    assert attempt["normalized_windows"] == {"left": [140, 212], "right": [454, 526]}
    assert attempt["raw_windows"] == {"left": [187, 285], "right": [621, 718]}
    assert normalize(SCENE_LEFT) == normalize(SCENE_RIGHT)[::-1]
    assert len(normalize(SCENE_LEFT)) == len(normalize(SCENE_RIGHT)) == 72
    assert attempt["paired_cursors_after"] == [72, 72]
    assert attempt["residuals"] == {"left": "", "right": ""}
    assert attempt["paired_trace"]["exact"] is True
    assert attempt["spacing_shell"]["single_boundary_spaces"] is True
    assert attempt["spacing_shell"]["double_spaces"] is False
    assert attempt["gates"]["connected_multi_event_scene"] is True
    assert attempt["gates"]["varied_predicates"] is True
    assert attempt["gates"]["complete_finite_clauses"] is True
    assert attempt["gates"]["no_duplicate_adjacent_roles"] is True


def test_full_parent_novelty_obstruction_is_persisted_without_promotion():
    payload = json.loads(OUT.read_text())
    row = payload["rows"][0]
    attempt = row["linked_scene_attempt"]
    assert row["promotion_status"]["promoted"] is False
    assert attempt["admission"]["exact_character_closure"] is True
    assert attempt["admission"]["accepted"] is False
    assert attempt["admission"]["exact_child_saved"] is False
    assert attempt["admission"]["obstruction"]["cursor"] == [72, 72]
    assert attempt["admission"]["obstruction"]["residual"] == {"left": "", "right": ""}
    assert attempt["reused_frames"]
    assert attempt["reused_clauses"]
    assert attempt["candidate_independent_audit"]["normalized_letters"] == 666
    assert attempt["candidate_independent_audit"]["two_pointer_exact"] is True
    assert len(attempt["candidate_sha256"]) == 64
    assert attempt["bounded_production_count"] == 1
    assert attempt["max_paired_scene_productions"] == 8
    assert row["next_operator"].startswith("change to a different actual seam")
