from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.endpoint_first_semantic_interior_20260913 import (
    InteriorSlot,
    SentencePlan,
    endpoint_pairs,
    independent_two_pointer,
    parse_surface_witness,
    prefix_debt,
    run,
    search_plan,
)


def test_endpoint_forms_are_intersected_before_interior() -> None:
    assert prefix_debt("the", "letter")["compatible"] is False
    shifted = prefix_debt("ab", "cba")
    assert shifted["compatible"] is True
    assert shifted["residual_debt"] == "c"
    plan = SentencePlan("fixture", "a fixture event", "Ab cba.", ("ab",),
                        (InteriorSlot("middle", ("",)),), ("cba",))
    rows = endpoint_pairs(plan)
    assert rows == [{"opening": "ab", "terminal": "cba",
                     "intersection": shifted, "compatible": True}]


def test_search_closes_shifted_boundaries_during_expansion() -> None:
    plan = SentencePlan("fixture", "a fixture event", "Ab cba.", ("ab",),
                        (InteriorSlot("middle", ("",)),), ("cba",))
    result = search_plan(plan, state_cap=100)
    assert [row["rendered"] for row in result["candidates"]] == ["Ab  cba."]
    assert independent_two_pointer(result["candidates"][0]["rendered"])["exact"]
    assert result["stats"]["partial_boundary_rejections"] == 0


def test_independent_witness_replays_complete_roles() -> None:
    plan = SentencePlan("fixture", "a fixture event", "Ab cba.", ("ab",),
                        (InteriorSlot("middle", ("",)),), ("cba",))
    assert parse_surface_witness(plan, "Ab  cba.")["independent_surface_parse"]
    assert not parse_surface_witness(plan, "Ab cba.")["independent_surface_parse"]


def test_bounded_run_records_rejections_and_keeps_reader_gate_closed() -> None:
    result = run(state_cap=2_000)
    assert result["config"]["endpoint_forms_selected_before_interior"]
    assert result["config"]["exact_closure_checked_during_search"]
    assert not result["config"]["catalogue_used_for_generation"]
    assert result["records"]
    assert result["mechanically_admitted"] == []
    for row in result["records"]:
        assert row["audit"]["sentence_witness"]["independent_surface_parse"]
        assert row["audit"]["current_central_admission"]
        assert row["audit"]["rejection_codes"]
        assert not row["audit"]["mechanically_admitted"]
        if row["kind"] == "partial_boundary_rejection":
            assert not row["audit"]["independent_two_pointer"]["exact"]


def test_source_controls_are_independently_replayable() -> None:
    result = run(state_cap=100)
    for plan in result["plans"]:
        assert plan["source_witness"]["independent_surface_parse"]
        assert plan["source_control"]["sentence_witness"]["independent_surface_parse"]
