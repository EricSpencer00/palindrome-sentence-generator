import json
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wfsa", ROOT / "experiments/character_wfsa_boundary_20260920.py")
wfsa = importlib.util.module_from_spec(spec); spec.loader.exec_module(wfsa)

def test_run_has_online_provenance_and_controls():
    out = wfsa.run()
    assert out["novelty_preflight"]["status"] == "passed"
    assert out["stats"]["prose_controls"] == 4
    assert out["provenance"]["independent_audit"]
    assert all(x["audit"]["sha256_forward"] != x["audit"]["sha256_reverse"] for x in out["prose_controls"])

def test_audit_independently_rejects_mutation():
    a = wfsa.audit("A sailor guides a harbor")
    assert a["letters"] == 20 and not a["pointer_exact"]
