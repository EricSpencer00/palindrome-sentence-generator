import json

from experiments.two_clause_character_csp_20260920 import OUT, audit, run

def test_independent_audit_is_pointer_and_hash_exact():
    checked = audit("An aide rips nine memos; some men inspire Diana.")
    assert checked["exact"] is True and checked["sha_equal"] is True
    assert checked["letters"] == 38

def test_csp_is_not_a_reverse_or_repair_lane():
    result = run()
    assert result["provenance"]["generated_compositionally"] is True
    assert result["provenance"]["novelty"].startswith("not seed-conditioned")
    assert result["reader_gate"].startswith("closed")
    OUT.write_text(json.dumps(result, indent=2) + "\n")
