from experiments.staggered_abba_paragraph_product_20260922 import (
    run,
    topology_control,
)


def test_formulaic_control_proves_staggered_topology_but_not_readability():
    control = topology_control()
    assert control["audit"]["two_pointer_exact"] is True
    assert control["audit"]["sentence_boundaries_staggered"] is True
    assert control["audit"]["aligned_internal_boundaries"] == []
    assert control["mechanically_admitted"] is False
    assert "control only" in control["status"]


def test_bounded_search_routes_only_admitted_closures_toward_readers():
    result = run(max_states=500, max_results=10)
    assert result["novelty_preflight"]["preclosed_sentence_units"] is False
    assert result["reader_packet"] == []
    assert all(row["mechanically_admitted"]
               for row in result["mechanically_admitted_candidates"])
