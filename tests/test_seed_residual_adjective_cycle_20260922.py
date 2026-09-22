import pytest

from experiments.seed_residual_adjective_cycle_20260922 import (
    CONTROL, pointer_audit, run,
)


@pytest.fixture(scope="module")
def result():
    return run(max_rows=50)


def test_repeated_word_control_is_exact_but_rejected(result):
    assert pointer_audit(CONTROL)["exact"] is True
    assert result["repeated_word_control"]["admitted"] is False
    assert result["repeated_word_control"]["mechanical_checks"]["distinct_words"] is False


def test_every_row_is_an_exact_nonempty_residual_cycle(result):
    for row in result["rows"]:
        assert row["cycle_equation"]["holds"] is True
        assert row["cycle_equation"]["intermediate_empty_residual"] is False
        assert row["independent_exact_audit"]["exact"] is True
        assert not set(row["left_adjectives"]) & set(row["right_adjectives"])
