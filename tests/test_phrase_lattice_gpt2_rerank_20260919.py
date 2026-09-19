from experiments.phrase_lattice_gpt2_rerank_20260919 import audit, novelty_preflight


def test_audit_independently_verifies_exact_candidate():
    row = audit("foe rut and i a sec nor even never once said nature of")
    assert row["letters"] == 42
    assert row["two_pointer_exact"] is True
    assert row["validator_exact"] is True
    assert row["sha256_forward"] == row["sha256_reverse"]


def test_preflight_records_this_lane_as_new():
    result = novelty_preflight()
    assert result["status"] == "passed"
    assert result["duplicate_sweep_rejected"] is False
    assert "posthoc-distilgpt2-rerank" in result["signature"]
