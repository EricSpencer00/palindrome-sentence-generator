from experiments.anchored_np_cycle_clause_search_20260922 import (
    ANCHORS,
    anchor_equation,
    run,
)


def test_natural_np_anchors_are_open_cycles_not_closed_pairs():
    for anchor in ANCHORS.values():
        equation = anchor_equation(anchor)
        assert equation["exact_open_cycle"] is True
        assert equation["start_debt"] == equation["end_debt"]
        assert equation["mid_debt"]
        assert equation["right_exposed"] != equation["left_exposed"]


def test_bounded_clause_probe_routes_only_central_admissions_to_readers():
    result = run(max_states_per_pair=200, max_results=10)
    assert result["stats"]["grammar_pairs"] == 8
    assert result["provenance"]["literal_cycle_pumping"] is False
    assert result["reader_packet"] == []
    assert all(row["mechanically_admitted"]
               for row in result["mechanically_admitted_candidates"])
