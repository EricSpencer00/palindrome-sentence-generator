from experiments.onset_conditioned_recursive_cfg_20261001 import audit, run, two_pointer


def test_audit_is_independent_and_exact_for_known_palindrome():
    row = audit("An aide rips nine memos; some men inspire Diana.")
    assert row["two_pointer_exact"] and row["validator_exact"]
    assert row["sha256_forward"] == row["sha256_reverse"]


def test_onset_product_records_real_prefilter_and_reader_gate():
    result = run()
    assert result["stats"]["raw_recursive_chart"] > 0
    # The current bounded chart exposes the intended dead-frontier evidence:
    # all surviving complete sentences begin with ``a`` while no complete
    # sentence in the independently generated chart ends in ``a``.
    assert "a" in result["stats"]["onset_classes"]
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["readability_gate"] == "closed; no human readers run, and exactness does not certify English prose"
