from experiments.incumbent_568_phrasewise_equation_audit_20260923 import (
    LEFT_PHRASE,
    RIGHT_PHRASE,
    run,
)


def test_rejected_child_is_exact_but_factors_at_a_phrasewise_seam():
    result = run()
    local = result["local_equation"]
    candidate = result["candidate"]

    assert local["exact_reverse_equation"] is True
    assert local["shared_reflected_boundaries"] == [7]
    assert local["anti_shortcut_pass"] is False
    assert [chunk["left"] for chunk in local["reverse_chunks"]] == [
        "redamar", "diaperedelena",
    ]
    assert candidate["letters"] == 578
    assert candidate["project_validator_exact"] is True
    assert candidate["independent_two_pointer_exact"] is True
    assert candidate["admission_status"].startswith("rejected:")
    assert result["provenance"]["new_insertions"] == [LEFT_PHRASE, RIGHT_PHRASE]


def test_audit_does_not_claim_readability_or_reader_evidence():
    result = run()
    assert result["candidate"]["reader_evidence"] is False
    assert result["provenance"]["reader_claim"] is False
    assert "not reader-tested" in result["candidate"]["readability_status"]
