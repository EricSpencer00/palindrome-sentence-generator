import importlib.util
from pathlib import Path

P = Path(__file__).parents[1] / "experiments/center_aware_slot_csp_20260921.py"
spec = importlib.util.spec_from_file_location("center_aware", P)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_seed_is_recovered_as_control_and_center_is_internal():
    result = m.run()
    assert len(result["controls"]) == 1
    control = result["controls"][0]
    assert control["audit"]["exact"] is True
    assert control["audit"]["letters"] == 38
    assert result["exact_gt38"] == []
    assert result["stats"]["states"] > 0

def test_audit_independently_checks_forward_reverse_hashes():
    a = m.audit("An aide rips nine memos; some men inspire Diana.")
    assert a["exact"] and a["sha_equal_under_reversal"]
    assert a["sha256_forward"] == a["sha256_reverse"]
