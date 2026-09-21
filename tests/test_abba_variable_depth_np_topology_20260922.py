from experiments.abba_variable_depth_np_topology_20260922 import audit, run

def test_seed_audit():
    assert audit("An aide rips nine memos; some men inspire Diana.")["two_pointer_exact"]

def test_variable_np_lane_records_fresh_controls_and_residuals():
    d=run()
    assert d["stats"]["subject_paths"] == 12
    assert d["stats"]["closed_derivations"] == 0
    assert d["stats"]["exact_gt38"] == 0
    assert len(d["controls"]) == 4
    assert all(not x["audit"]["two_pointer_exact"] for x in d["controls"])
    assert all(x["residual_prefix"] for x in d["residual_certificates"])
