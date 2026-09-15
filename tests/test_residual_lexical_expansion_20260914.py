from experiments.dual_plan_residual_search_20260914 import PLANS
from experiments.residual_lexical_expansion_20260914 import (
    compatible_emission,
    mine_additions,
    trace_pair,
)


def test_trace_records_live_role_and_orientation():
    left = next(plan for plan in PLANS if plan.name == "numbered_svo")
    right = next(plan for plan in PLANS if plan.name == "plural_name")
    rows = trace_pair(left, right)
    assert rows
    assert all(row["owner"] in (-1, 1) and row["residual"] for row in rows)
    assert any(row["next_role"] == "name" for row in rows)


def test_mined_words_are_prefix_compatible_with_their_witness():
    traces = [{"left_plan": "x", "right_plan": "y", "left_index": 0,
               "right_index": 0, "residual": "aid", "owner": 1,
               "next_role": "name", "matched_letters": 0}]
    additions, witnesses = mine_additions(traces)
    for row in witnesses:
        assert compatible_emission(row["residual"], row["candidate"],
                                   row["owner"]) == row["residual_after"]
    assert isinstance(additions, dict)


def test_incompatible_emission_is_rejected():
    assert compatible_emission("abc", "ordinary", 1) is None
