from terminal_bigram_abba_operator_20260921 import run


def test_terminal_bigram_gate_is_fail_closed():
    result = run()
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["bank"] == 64
    assert result["stats"]["eligible_state_count"] == 0
    assert result["stats"]["exact_reader_candidates"] == 0
    assert result["reader_facing_eligibility"]["decision"] == "ineligible"
    assert "no complete sentence tuple" in result["reader_facing_eligibility"]["reason"]
