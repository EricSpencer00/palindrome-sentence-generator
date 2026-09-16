from experiments.exact_tape_semantic_slot_repair_20260916 import (
    EXPERIMENT,
    PUNCTUATION,
    SOURCE_TAPE,
    exact_slice,
    exact_two_pointer,
    hash_audit,
    novelty_preflight,
    search,
)


def test_independent_exact_hash_controls():
    for text, expected in (("A new sentence.", False), ("An aide rips nine memos; Some men inspire Diana.", True)):
        assert exact_slice(text)["exact"] is expected
        assert exact_two_pointer(text)["exact"] is expected
        assert hash_audit(text)["exact"] is expected


def test_search_uses_existing_exact_tape_and_typed_repairs():
    assert novelty_preflight()["passed"]
    result = search()
    assert result["experiment"] == EXPERIMENT
    assert 0 < result["states_examined"] < (3 ** 6 - 1) * len(PUNCTUATION) * 2
    assert result["source_evidence"]["source_exact_and_mechanically_checked"]
    assert result["source_evidence"]["source_tape"] == SOURCE_TAPE
    assert result["tape_preserved_count"] == 0
    assert result["exact_count"] == 0
    assert result["mechanically_admitted_count"] == 0
    assert all(row["heldout_repair"]["changed_slot_count"] == 1 for row in result["failed_attempts"])
    assert result["independent_exact_agreement_count"] == result["states_examined"]
    assert result["admission_agreement_count"] == result["states_examined"]
    assert all(row["operation"] != "identity" and row["letters"] > len(SOURCE_TAPE) for row in result["failed_attempts"])


def test_best_probes_are_ordinary_order_and_not_wrappers():
    result = search()
    for row in result["best_rendered_candidates"]:
        assert row["provenance"]["wrapper_used"] is False
        assert row["provenance"]["word_order_mirrored"] is False
        assert row["rendered"].strip().endswith(".")
