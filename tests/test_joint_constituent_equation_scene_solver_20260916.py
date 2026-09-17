from experiments.joint_constituent_equation_scene_solver_20260916 import (
    EXPERIMENT, LEFT, RIGHT, exact_sha, exact_two_pointer, novelty_preflight, search,
)


def test_independent_audits_agree_on_known_controls():
    assert exact_two_pointer("Able was I ere I saw Elba.")["exact"]
    assert exact_sha("Able was I ere I saw Elba.")["exact"]
    assert not exact_two_pointer("The careful clerk records the manifest.")["exact"]


def test_novelty_preflight_and_joint_constituent_search_are_bounded():
    assert novelty_preflight()["passed"]
    result = search()
    assert result["experiment"] == EXPERIMENT
    assert result["states_examined"] == (len(LEFT) * (len(LEFT) - 1)) * (len(RIGHT) * (len(RIGHT) - 1))
    assert result["exact_count"] == 0
    assert len(result["best_rendered_candidates"]) == 9
    assert all(row["anti_shortcut_flags"]["complete_constituents_only"] for row in result["best_rendered_candidates"])
    assert all(row["independent_exact_agreement"] for row in result["best_rendered_candidates"])
