import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location("lane", Path(__file__).with_name("paragraph_residual_state_dp_20260921.py"))
lane=importlib.util.module_from_spec(spec); spec.loader.exec_module(lane)

def test_bounded_run_tracks_grammar_roles_and_live_prunes():
    result=lane.run()
    assert result["stats"]["rendered"] == 16
    assert result["stats"]["residual_prunes"] == 16
    assert result["stats"]["exact_gt38"] == 0
    assert all(row["grammar_state"]["discourse"] == "ABBA" for row in result["rendered_controls"])
    assert all(row["provenance"]["post_hoc_repair"] is False for row in result["rendered_controls"])

def test_exact_gate_is_independent_and_hashes_disagree_for_controls():
    result=lane.run()
    for row in result["rendered_controls"]:
        assert row["audit"]["pointer_exact"] is False
        assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]

