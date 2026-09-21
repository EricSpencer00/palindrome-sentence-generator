from pathlib import Path
from experiments.cadence_grammar_orbit_20260920 import audit, novelty_preflight, run

def test_run_has_live_orbit_and_controls():
    r=run()
    assert novelty_preflight()["status"] == "passed"
    assert r["stats"]["obligations_checked"] > 0
    assert r["stats"]["prose_controls"] >= 3
    assert r["provenance"]["catalogue_text"] is False
    assert Path("runs/cadence-grammar-orbit-20260920.json").exists()

def test_independent_audit_rejects_plain_clause():
    a=audit("At dawn the patient ferryman marks the harbor bell.")
    assert not a["pointer_exact"]
    assert a["sha256_forward"] != a["sha256_reverse"]
