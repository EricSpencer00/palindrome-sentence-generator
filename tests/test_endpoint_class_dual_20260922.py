from experiments.endpoint_class_dual_20260922 import run


def test_zero_output_probe_does_not_claim_novelty_or_reader_progress():
    result = run(max_states=100)
    assert result["stats"]["exact_admitted"] == 0
    assert result["reader_candidates"] == []
    assert result["novelty_preflight"]["status"] == "not_recorded"
    assert result["novelty_preflight"]["registry_inspected"] is False
