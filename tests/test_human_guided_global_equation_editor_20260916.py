import json

from experiments.human_guided_global_equation_editor_20260916 import (
    CLAUSES,
    EXPERIMENT,
    PUNCTUATION,
    SCENE,
    exact_by_slice,
    exact_by_two_pointer,
    novelty_preflight,
    render_with_tense,
    search,
)


def test_scene_is_long_and_global_state_changes_all_clauses():
    assert len("".join(c for c in SCENE["seed"].lower() if c.isalpha())) >= 100
    first, first_state = render_with_tense((0, 0, 0, 0, "comma", "past"))
    second, second_state = render_with_tense((2, 2, 2, 2, "dash", "present"))
    assert first != second
    assert first_state["tense"] != second_state["tense"]
    assert first_state["punctuation_policy"] != second_state["punctuation_policy"]
    assert all(word in second for word in ("unlatches", "examines", "describes", "misses"))


def test_independent_exact_checks_agree_on_controls():
    for text, expected in (("A quiet room.", False), ("Able was I ere I saw Elba.", True)):
        assert exact_by_slice(text)["exact"] is expected
        assert exact_by_two_pointer(text)["exact"] is expected


def test_preflight_and_actual_search_retain_best_candidates():
    preflight = novelty_preflight()
    assert preflight["passed"]
    result = search()
    assert result["experiment"] == EXPERIMENT
    assert result["states_examined"] == 3 ** 4 * 2 * len(PUNCTUATION)
    assert 0 < len(result["best_rendered_candidates"]) <= 24
    assert result["independent_exact_agreement_count"] == result["states_examined"]
    assert result["readability_evidence"]["reader_eligible_count"] == 0
    assert result["provenance"]["pre_existing_palindrome_wrapped"] is False
