import json
from experiments.typed_finite_suffix_20260930 import audit, run

def test_audit_independent_control():
    row = run()["structural_control"]
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["sha_equal"]
    assert row["audit"]["letters"] == 54

def test_finite_lane_has_no_nominal_suffix_and_records_residuals():
    result = run()
    assert all("Now, an aid" not in x["text"] for x in result["rendered_candidates"])
    assert result["stats"]["live_pairs"] == 30
    assert result["residuals"]
    assert result["reader_gate"].startswith("closed")
