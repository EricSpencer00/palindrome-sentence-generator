import pytest

from experiments.seed_open_residual_discourse_join_20260922 import (
    INCUMBENT, _pointer_audit, run,
)


@pytest.fixture(scope="module")
def result():
    return run(maximum_rows=50)


def test_incumbent_remains_an_exact_control_only(result):
    assert _pointer_audit(INCUMBENT)["exact"] is True
    controls = [row for row in result["rows"] if row["incumbent_control"]]
    assert controls
    assert all(not row["mechanically_admitted"] for row in controls)


def test_every_promoted_row_is_cross_sentence_and_independently_exact(result):
    for row in result["mechanically_admitted_candidates"]:
        assert row["independent_exact_audit"]["exact"] is True
        assert row["structural_audit"]["cross_sentence_coupled"] is True
        assert row["has_two_generated_inner_sentences"] is True
