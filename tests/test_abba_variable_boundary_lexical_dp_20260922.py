from experiments.abba_variable_boundary_lexical_dp_20260922 import audit, lexical_dp, run

def test_audit_independently_detects_seed_and_nonpalindrome():
    assert audit("An aide rips nine memos; some men inspire Diana.")["two_pointer_exact"]
    assert not audit("A careful gardener waters the orchard.")["two_pointer_exact"]

def test_dp_requires_whole_obligation_and_exposes_frontier():
    parses, frontier = lexical_dp("thepatientarchivist")
    assert parses == []
    assert frontier

def test_artifact_records_fresh_abba_controls_and_exact_gate():
    data = run()
    assert data["stats"]["closed_derivations"] == 0
    assert data["stats"]["exact_gt38"] == 0
    assert len(data["controls"]) == 2
    assert all(c["audit"]["two_pointer_exact"] is False for c in data["controls"])
    branches = data["heldout_repair"]["branches"]
    assert {b["branch"] for b in branches} == {"red", "elc"}
    assert all(b["parse_count"] == 0 for b in branches)
    assert any(b["branch"] == "elc" and b["ordinary_english_certificate"] for b in branches)
    assert any(b["branch"] == "red" and b["deepest_matched_prefix"] >= 3 for b in branches)
