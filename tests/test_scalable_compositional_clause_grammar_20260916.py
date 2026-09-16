from experiments.scalable_compositional_clause_grammar_20260916 import (
    BASE,
    EXPERIMENT,
    TARGET_BANDS,
    exact_slice,
    exact_two_pointer,
    hash_audit,
    novelty_preflight,
    search,
)


def test_fresh_compositional_base_is_ordinary_and_not_catalogue_text():
    assert "gardener" in BASE
    assert "Marge" not in BASE
    assert len(BASE) > 40


def test_independent_exact_hash_controls_agree():
    for text, expected in (("The garden grows.", False), ("Able was I ere I saw Elba.", True)):
        assert exact_slice(text)["exact"] is expected
        assert exact_two_pointer(text)["exact"] is expected
        assert hash_audit(text)["exact"] is expected


def test_length_indexed_search_emits_each_target_and_repairs_every_miss():
    assert novelty_preflight()["passed"]
    result = search()
    assert result["experiment"] == EXPERIMENT
    assert result["maximum_compositional_depth_searched"] == 5
    assert result["states_examined"] > 0
    assert len(result["target_results"]) == len(TARGET_BANDS)
    assert all(item["rendered_probes"] for item in result["target_results"])
    assert all(row["next_repair_operator"] for row in result["failed_attempts"])
    assert all(row["provenance"]["proper_multiword_palindromic_span_forbidden"] for row in result["best_rendered_candidates"])
