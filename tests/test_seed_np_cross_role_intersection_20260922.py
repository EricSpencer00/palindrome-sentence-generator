import pytest

from experiments.seed_np_cross_role_intersection_20260922 import run


@pytest.fixture(scope="module")
def result():
    return run(max_rows=200)


def test_reader_candidate_is_exact_and_mechanically_admitted(result):
    candidates = result["reader_study_candidates"]
    assert len(candidates) == 1
    row = candidates[0]
    assert row["rendered"] == (
        "An aide rips nine memo-hero memos. "
        "Some more home men inspire Diana."
    )
    assert row["independent_exact_audit"]["letters"] == 54
    assert row["independent_exact_audit"]["two_pointer_exact"] is True
    assert row["independent_exact_audit"]["project_validator"] is True
    assert row["independent_exact_audit"]["hashes_agree"] is True
    assert row["mechanically_admitted"] is True
    assert all(row["mechanical_checks"].values())


def test_cross_role_boundary_and_provenance_are_real(result):
    row = result["reader_study_candidates"][0]
    assert row["live_equation"]["holds"] is True
    assert row["live_equation"]["intermediate_empty_residual"] is False
    assert row["boundary_audit"]["different_segmentation"] is True
    assert row["boundary_audit"]["aligned_internal_boundaries"] == []
    assert row["provenance"]["finished_tape_reversal"] is False
    assert row["provenance"]["per_candidate_rlaif"] is False
    assert result["reader_results"] == []
