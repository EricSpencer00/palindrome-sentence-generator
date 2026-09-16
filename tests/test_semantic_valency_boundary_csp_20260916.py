from experiments.semantic_valency_boundary_csp_20260916 import (
    DELIMITERS,
    EXPERIMENT,
    SCENE,
    exact_slice,
    exact_two_pointer,
    novelty_preflight,
    render,
    search,
)


def test_fresh_station_scene_and_sense_frames_are_not_archive_seed():
    assert "station porter" in SCENE["seed"]
    assert "archivist" not in SCENE["seed"]
    assert len("".join(character for character in SCENE["seed"] if character.isalpha())) >= 100


def test_exact_audits_agree_independently():
    for text, expected in (("Quiet work continues.", False), ("Able was I ere I saw Elba.", True)):
        assert exact_slice(text)["exact"] is expected
        assert exact_two_pointer(text)["exact"] is expected


def test_preflight_and_search_retain_failures_and_repair_operator():
    assert novelty_preflight()["passed"]
    result = search()
    assert result["experiment"] == EXPERIMENT
    assert result["states_examined"] == 4 ** 3 * 2 * len(DELIMITERS)
    assert len(result["best_rendered_candidates"]) == 24
    assert len(result["failed_attempts"]) == result["states_examined"]
    assert all(row["next_repair_operator"] for row in result["failed_attempts"])
    assert result["independent_exact_agreement_count"] == result["states_examined"]
    assert result["readability_evidence"]["reader_eligible_count"] == 0
