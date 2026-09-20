from experiments.centerout_lexical_scene_lattice_20260920 import run

def test_centerout_lexical_lattice_has_live_audit_and_provenance():
    result = run()
    assert result["candidate_count"] == 27
    assert result["provenance"]["source_catalogues_used"] == []
    for row in result["candidates"]:
        assert row["audit"]["sha_equal_under_reversal"] == row["audit"]["independent_two_pointer_exact"]
        assert row["center_out"]["growth_order"] == ["center", "left", "right"]
        assert row["provenance"]["human_authored_alternatives"]
