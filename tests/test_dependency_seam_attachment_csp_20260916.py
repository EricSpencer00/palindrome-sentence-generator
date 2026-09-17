from experiments.dependency_seam_attachment_csp_20260916 import ATTACHMENTS, EVENTS, SUBJECTS, preflight, run


def test_dependency_attachment_csp_is_new_and_independently_checked():
    result = run()
    assert result["novelty_preflight"]["passed"]
    assert result["states_examined"] == 64
    assert result["exact_count"] == 0
    assert len(result["rendered_candidates"]) == 64
    for row in result["rendered_candidates"][:8]:
        assert row["rendered"]
        assert row["independent_reparse"]
        assert row["independent_exact_agreement"]
        assert row["anti_shortcut_flags"]["complete_dependency_constituent"]
        assert row["next_repair"]


def test_inventory_is_joint_subject_event_attachment_space():
    assert [len(x) for x in (SUBJECTS, EVENTS, ATTACHMENTS)] == [4, 4, 4]
    assert preflight()["collisions"] == []
