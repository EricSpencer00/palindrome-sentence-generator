import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOD = runpy.run_path(str(ROOT / "experiments/shared_noun_active_passive_20260930.py"))


def test_audit_independent_pointer_and_sha():
    text = "The sailor guards the lantern; the lantern was guarded by the sailor."
    au = MOD["audit"](text)
    assert au["letters"] > 0
    assert au["sha_equal"] is False
    assert MOD["pointer_exact"](text) is False


def test_run_writes_online_provenance_artifact(tmp_path, monkeypatch):
    result = MOD["run"](state_limit=20000)
    assert result["method"].startswith("online shared-patient")
    assert result["novelty_preflight"]["passed"] is True
    assert result["independent_validation"] == ["literal outside-in two-pointer", "forward/reverse SHA-256"]
    assert all(c["reader_eligible"] is False for c in result["controls"])
