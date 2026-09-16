from experiments.reader_first_discourse_scene_lattice_20260916 import (
    EXPERIMENT,
    JOINERS,
    SCENES,
    TENSE_PATTERNS,
    exact_slice,
    exact_two_pointer,
    novelty_preflight,
    search,
)


def test_reader_first_seeds_are_complete_and_topic_continuous():
    assert len(SCENES) == 2
    for scene in SCENES:
        assert len(scene.seed) > 100
        assert len(scene.clauses) == 3
        assert len(scene.topics) == 3
        assert "it" in scene.seed


def test_independent_exact_controls_agree():
    for text, expected in (("The audience rests.", False), ("Able was I ere I saw Elba.", True)):
        assert exact_slice(text)["exact"] is expected
        assert exact_two_pointer(text)["exact"] is expected


def test_search_logs_failed_states_and_single_slot_repairs():
    assert novelty_preflight()["passed"]
    result = search()
    assert result["experiment"] == EXPERIMENT
    assert result["states_examined"] == 2 * 3 ** 3 * len(TENSE_PATTERNS) * len(JOINERS)
    assert len(result["best_rendered_candidates"]) == 24
    assert len(result["failed_attempts"]) == result["states_examined"]
    assert all(row["bounded_repair"]["changed_slot_count"] == 1 for row in result["failed_attempts"])
    assert result["independent_exact_agreement_count"] == result["states_examined"]
    assert result["independent_admission_agreement_count"] == result["states_examined"]
    assert result["readability_evidence"]["reader_eligible_count"] == 0
