from experiments.inflectional_clitic_boundary_csp_20260916 import (
    EXPERIMENT,
    PUNCTUATION,
    exact_slice,
    exact_two_pointer,
    hash_audit,
    novelty_preflight,
    search,
)


def test_independent_exact_and_hash_controls_agree():
    for text, expected in (("Harbor work continues.", False), ("Able was I ere I saw Elba.", True)):
        assert exact_slice(text)["exact"] is expected
        assert exact_two_pointer(text)["exact"] is expected
        assert hash_audit(text)["exact"] is expected


def test_preflight_and_search_retain_heldout_repairs():
    assert novelty_preflight()["passed"]
    result = search()
    assert result["experiment"] == EXPERIMENT
    assert result["states_examined"] == 2 * 2 * 3 * 2 * 2 * 2 * len(PUNCTUATION)
    assert len(result["best_rendered_candidates"]) == 24
    assert len(result["failed_attempts"]) == result["states_examined"]
    assert all(row["heldout_repair"]["changed_slot_count"] == 1 for row in result["failed_attempts"])
    assert result["independent_exact_agreement_count"] == result["states_examined"]
    assert result["admission_agreement_count"] == result["states_examined"]


def test_run_is_complete_ordinary_order_and_not_a_wrapper():
    result = search()
    for row in result["best_rendered_candidates"]:
        assert row["rendered"].endswith(".")
        assert row["provenance"]["pre_existing_palindrome_wrapped"] is False
        assert row["provenance"]["word_order_mirrored"] is False
