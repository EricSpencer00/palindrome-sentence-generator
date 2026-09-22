import pytest

from experiments.seed_residual_modifier_phrase_cycle_20260922 import run


@pytest.fixture(scope="module")
def result():
    return run(max_rows=50)


def test_distinct_modifier_cycles_are_independently_exact(result):
    for row in result["rows"]:
        assert row["cycle_equation"]["holds"] is True
        assert row["cycle_equation"]["intermediate_empty_residual"] is False
        assert row["independent_exact_audit"]["exact"] is True
        assert not set(row["left_modifier"]) & set(row["right_modifier"])


def test_inventory_and_provenance_are_recorded(result):
    assert result["stats"]["attested_modifier_phrases"] > 0
    assert result["novelty_preflight"]["not_a_larger_adjective_sweep"] is True
