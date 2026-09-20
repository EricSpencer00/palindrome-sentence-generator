import json
from pathlib import Path
from experiments.character_clause_trie_csp_20260920 import audit, run

def test_character_csp_has_independent_audits_and_complete_controls():
    result = run()
    assert result["signature"].startswith("character-csp|")
    assert result["stats"]["left_clauses"] == 6
    assert result["stats"]["right_clauses"] == 6
    assert result["stats"]["fresh_exact_gt38"] == 0
    assert all(x["audit"]["independent_two_pointer"] is False for x in result["controls"])
    assert all(x["provenance"]["independent_clause_banks"] for x in result["rendered_candidates"])

def test_audit_detects_exact_tape_independently():
    a = audit("An aide rips nine memos; some men inspire Diana.")
    assert a["exact"] and a["independent_two_pointer"]
    assert a["sha256_forward"] == a["sha256_reverse"]
