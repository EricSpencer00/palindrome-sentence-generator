from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.constituent_alignment_repair_20260912 import (  # noqa: E402
    DEFAULT_PROPOSALS_PER_PLAN,
    MAX_LETTERS,
    MIN_LETTERS,
    PLANS,
    audit_proposal,
    independent_two_pointer,
    paired_constituent_variants,
    run,
)


def test_two_pointer_is_independent_and_finds_first_mismatch() -> None:
    exact = independent_two_pointer("No lemon, no melon.")
    assert exact["exact"] is True
    assert exact["letters"] == 14
    mismatch = independent_two_pointer("Careful editors.")
    assert mismatch["exact"] is False
    assert mismatch["first_mismatch"] == [0, mismatch["letters"] - 1]


def test_lattice_records_whole_constituent_insertions_and_deletions() -> None:
    plan = PLANS[0]
    proposals = paired_constituent_variants(plan, limit=DEFAULT_PROPOSALS_PER_PLAN)
    paired = next(row for row in proposals if row["operator"] == "paired_insert_delete_constituents")
    operations = paired["operations"]
    assert {operation["operation"] for operation in operations} == {
        "delete_constituent", "insert_constituent"
    }
    assert all("text" in operation and "role" in operation for operation in operations)
    assert all(not operation["operation"].endswith("character") for operation in operations)


def test_source_clauses_are_not_derived_by_reversal() -> None:
    assert all(plan.left_source != plan.right_source[::-1] for plan in PLANS)


def test_exact_fixture_is_a_rendered_proposal_but_not_readability_claim() -> None:
    # This fixture exercises the exact audit only; it is deliberately short and
    # would not be a prospective reader item under the experiment's length band.
    assert audit_proposal("No lemon, no melon.")["independent_two_pointer"]["exact"]
    assert audit_proposal("No lemon, no melon.")["mechanically_admitted"] is False


def test_bounded_run_records_every_proposal_and_keeps_reader_gate_closed() -> None:
    result = run(proposals_per_plan=12)
    assert result["config"]["plans"] == len(PLANS) == 6
    assert result["config"]["proposals_per_plan"] == 12
    assert len(result["records"]) == 72
    assert len(result["records"]) == sum(
        len(paired_constituent_variants(plan, limit=12)) for plan in PLANS
    )
    assert all("source_clauses" in row for row in result["records"])
    assert all("audit" in row and "independent_two_pointer" in row["audit"] for row in result["records"])
    assert all("current_central_admission" in row["audit"] for row in result["records"])
    assert result["readable_survivors"] == []
    assert MIN_LETTERS == 100
    assert MAX_LETTERS >= MIN_LETTERS
