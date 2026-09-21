from experiments.seed_full_residual_joint_frames_20260921 import audit, run

def test_bounded_joint_frames_are_complete_and_audited():
    result = run()
    assert result["novelty_preflight"]["status"] == "passed"
    assert len(result["rows"]) == 4
    assert result["stats"]["full_openings"] >= 2
    for row in result["rows"]:
        assert row["grammar"]["left_complete"] and row["grammar"]["right_complete"]
        assert audit(row["rendered"]) == row["audit"]
        assert row["provenance"]["joint_semantic_frame_selection"]
        assert row["provenance"]["finished_tape_reversal"] is False

