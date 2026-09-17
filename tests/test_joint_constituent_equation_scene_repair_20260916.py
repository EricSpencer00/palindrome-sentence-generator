from experiments.joint_constituent_equation_scene_repair_20260916 import (
    EXPERIMENT,
    audit,
    novelty_preflight,
    run,
)


def test_targeted_repair_preflights_before_render_and_preserves_scope():
    result = run()
    assert result["experiment"] == EXPERIMENT
    assert result["novelty_preflight"]["passed"]
    assert result["candidate_count"] == 1
    row = result["candidate"]
    assert row["letters"] >= 90
    assert row["exact_check_two_pointer"]["exact"] is False
    assert row["exact_check_sha256"]["exact"] is False
    assert row["independent_exact_agreement"]
    assert row["anti_shortcut_flags"]["complete_constituents_only"]
    assert row["anti_shortcut_flags"]["isolated_character_edit"] is False
    assert row["anti_shortcut_flags"]["normal_order_events"]
    assert row["replaced_pair"]["left"]["position"] == "first"
    assert row["replaced_pair"]["right"]["position"] == "last"
    assert row["next_repair"]


def test_repair_audit_is_reproducible_and_reports_provenance():
    first, second = audit(), audit()
    assert first == second
    assert first["provenance"]["repair_operator"].startswith("paired complete")
    assert novelty_preflight()["passed"]
