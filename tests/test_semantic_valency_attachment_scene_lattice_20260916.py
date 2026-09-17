from experiments.semantic_valency_attachment_scene_lattice_20260916 import (
    CLAUSES, EXPERIMENT, novelty_preflight, run
)


def test_joint_scene_lattice_is_bounded_and_independently_audited():
    result = run()
    assert result["experiment"] == EXPERIMENT
    assert result["novelty_preflight"]["passed"]
    assert result["states_examined"] == 27
    assert result["exact_count"] == 0
    assert result["best_rendered_candidates"]
    for row in result["best_rendered_candidates"]:
        assert row["rendered"]
        assert row["independent_exact_agreement"]
        assert row["anti_shortcut_flags"]["semantic_valency_checked"]
        assert row["anti_shortcut_flags"]["complete_constituents_only"]
        assert row["next_repair"]


def test_preflight_is_done_against_new_signature():
    preflight = novelty_preflight()
    assert preflight["passed"]
    assert preflight["exact_signature_collisions_before_render"] == []
    assert [len(domain) for domain in CLAUSES] == [3, 3, 3]
