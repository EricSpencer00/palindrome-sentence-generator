import json
from pathlib import Path

from experiments.freshness_register_morphology_cycle_20260922 import (
    CYCLE_SCHEDULE,
    OUT,
    RESIDUAL,
    independent_audit,
    run,
)


def test_every_depth_is_an_exact_fresh_nonempty_residual_child():
    data = run()
    assert [row["depth"] for row in data["children"]] == [1, 2, 3, 4, 5]
    for row in data["children"]:
        assert row["mechanically_admitted"] is True
        assert all(row["mechanical_checks"].values())
        assert row["independent_audit"]["two_pointer_exact"] is True
        assert row["independent_audit"]["hashes_agree"] is True
        assert row["freshness"]["all_lemmas_distinct"] is True
        assert row["freshness"]["carrier_content_disjoint"] is True
        assert row["boundary_mask"]["passed"] is True
        assert row["stacked_equation"]["holds"] is True
        assert all(cycle["holds"] for cycle in row["cycles_inner_first"])
        assert all(cycle["residual_nonempty"] for cycle in row["cycles_inner_first"])
        assert all(step["residual_before"] == RESIDUAL for step in row["register_trace"])
        assert all(step["residual_after"] == RESIDUAL for step in row["register_trace"])


def test_k1_and_k2_are_the_preserved_interpretable_witnesses():
    rows = run()["children"]
    assert rows[0]["rendered"] == "No trace. Note: Spoons snoop. Set one carton."
    assert rows[0]["independent_audit"]["sha256_forward"] == (
        "d49d83577cbca30d96a2156901e8f5dbebd57ebbed304e18d39a6fc1e6e7bfe0"
    )
    assert rows[1]["rendered"] == (
        "No trace. Note: Spot spoons. Snoop. Stop. Set one carton."
    )
    assert rows[1]["independent_audit"]["sha256_forward"] == (
        "1013004f658bdefeaaf7dea69c6d90a5d2c53381dbb5d7290ab7b25a8e5de1c3"
    )
    assert all(row["syntax"]["direct_status"] == "intact_interpretable"
               for row in rows[:2])


def test_k3_through_k5_extend_the_same_lifo_branch_and_record_obstruction():
    data = run()
    for row in data["children"]:
        depth = row["depth"]
        expected = [cycle.y for cycle in CYCLE_SCHEDULE[:depth]]
        assert row["stack_pop_order"] == expected
        assert len(row["register_trace"]) == depth
    assert data["stats"]["syntax_obstruction_depths"] == [3, 4, 5]
    assert data["obstruction"]["first_depth"] == 3
    assert "inside T(y)" in data["obstruction"]["next_productive_morphology_operator"]
    assert data["reader_gate"]["status"] == "closed"


def test_fresh_carrier_pairs_are_rendered_audited_and_ranked_for_continuity():
    preflight = run()["carrier_preflight"]
    assert preflight["common_word_pairs_tested"] == 5
    assert sum(row["selected"] for row in preflight["rows"]) == 1
    for row in preflight["rows"]:
        assert row["equation"]["holds"] is True
        assert row["independent_audit"]["two_pointer_exact"] is True
        assert row["independent_audit"]["hashes_agree"] is True
        assert row["boundary_mask_passed"] is True


def test_independent_audit_rejects_a_nonpalindrome():
    assert independent_audit("Spoons snoop.")["two_pointer_exact"] is False


def test_compact_checked_in_artifact_matches_generator():
    assert OUT.exists()
    assert json.loads(Path(OUT).read_text()) == run()
