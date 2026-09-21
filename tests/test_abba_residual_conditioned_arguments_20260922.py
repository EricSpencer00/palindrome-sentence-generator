import importlib.util
from pathlib import Path

P = Path(__file__).parents[1] / "experiments/abba_residual_conditioned_arguments_20260922.py"
spec = importlib.util.spec_from_file_location("lane", P)
lane = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lane)

def test_run_has_independent_audits_and_residual_conditioning():
    data = lane.run()
    assert data["stats"]["frames"] == 3
    assert data["stats"]["exact_gt38"] == 0
    assert all(c["selected_residual_key"] for c in data["residual_certificates"])
    assert all("sha256_forward" in row["audit"] for row in data["controls"])
    assert data["novelty_preflight"]["finished_tape_reversal"] is False

def test_audit_rejects_near_miss():
    assert lane.audit("A sentence")["two_pointer_exact"] is False
    exact = "A man, a plan, a canal: Panama"
    assert lane.audit(exact)["two_pointer_exact"] is True
