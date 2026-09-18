import importlib

M = importlib.import_module("experiments.cross_attachment_resegmentation_20260918")

def test_controls_are_intact_and_independently_audited():
    rows = M.controls()
    assert rows and all(r["audit"]["letters"] >= 40 for r in rows)
    assert all("sha256_forward" in r["audit"] and "sha256_reverse" in r["audit"] for r in rows)
    assert all(r["provenance"]["catalogue_used"] is False for r in rows)

def test_resegmentation_lane_records_live_attachment_equations():
    out = M.run()
    assert out["fresh_exact_closures"] == []
    assert out["construction"]["movable_attachment_boundary"]
    assert out["construction"]["reciprocal_valency"]
    assert out["reader_gate"]["status"] == "not_triggered"

def test_independent_audit_rejects_non_palindrome():
    assert M._ind("The baker marks a map near dawn") ["is_palindrome"] is False
