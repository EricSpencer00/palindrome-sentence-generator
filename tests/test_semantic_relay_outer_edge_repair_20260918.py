from experiments.semantic_relay_outer_edge_repair_20260918 import run


def test_outer_edge_repair_is_bounded_and_keeps_reader_gate_closed():
    result = run()
    assert result["stats"]["probes"] == 8
    assert result["stats"]["exact"] == 0
    assert result["stats"]["mechanically_admitted"] == 0
    assert result["stats"]["reader_eligible"] == 0


def test_outer_edge_repair_improves_first_character_frontier():
    result = run()
    assert result["stats"]["best_matched_outer_characters"] >= 2
    assert result["best_frontier"]["provenance"]["operator"].startswith("outer-edge")
