from experiments.corrected_paired_clause_cfg_20260921 import audit, pair_clause, run

def test_acceptance_is_full_tape_not_left_arm():
    row = pair_clause("The baker marks a map.", "A clerk keeps the memo.")
    assert row["provenance"]["self_palindromic_halves_required"] is False
    assert row["left_half_palindromic"] is False
    assert not row["audit"]["pointer_exact"]

def test_audit_hashes_and_word_boundaries_are_independent():
    row = pair_clause("A nurse carries the chart.", "The writer finds a note.")
    assert row["word_boundaries"]
    assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]
    assert row["bilateral_trace"][0]["obligation"] == "cross-arm-equal"

def test_run_has_controls_and_provenance():
    result = run()
    assert result["stats"]["pairs"] == 16
    assert result["controls"]
    assert result["novelty_preflight"]["status"] == "passed"
