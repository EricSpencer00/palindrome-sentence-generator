from experiments.dream_rsi_masked_infilling_20260917 import audit, letters, run


def test_independent_audit_detects_exact_and_mismatch():
    assert audit("Step on, no pets.")["two_pointer_exact"]
    assert not audit("A patient nurse checks each dosage.")["two_pointer_exact"]
    assert audit("Step on, no pets.")["sha256_forward"] == audit(
        "Step on, no pets."
    )["sha256_reverse"]


def test_masked_infilling_run_records_fresh_provenance_and_repair():
    result = run(rounds=2, beam=4)
    assert result["stats"]["rendered"] > 0
    assert result["stats"]["exact"] == len(result["exact_candidates"])
    assert result["failure_and_repair"]["next_repair"]
    assert all(row["provenance"].get("fresh_authored_scene") for row in result["rows"])
    assert all("audit" in row and "rendered" in row for row in result["rows"])
