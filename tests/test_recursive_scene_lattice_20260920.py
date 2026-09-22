from experiments.recursive_scene_lattice_20260920 import audit, run


def test_audit_detects_exact_and_mismatch():
    assert audit("A man, a plan, a canal: Panama.")["exact"]
    assert not audit("The harbor pilot charts.")["exact"]


def test_recursive_lane_emits_prose_controls_and_live_traces():
    result = run()
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["recursive_states"] == result["stats"]["rendered_controls"]
    assert result["diagnostic_controls"]
    assert all(row["rendered"].endswith(".") for row in result["diagnostic_controls"])
    assert all("finished_tape_reversal" in row["provenance"] for row in result["diagnostic_controls"])
