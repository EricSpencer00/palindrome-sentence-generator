from experiments.paragraph_abba_dialogue_topology_20260926 import audit, run

def test_dialogue_topology_has_complete_controls_and_independent_audits():
    data = run()
    assert data["stats"]["branches"] == 4
    assert data["controls"] and all(c["audit"]["letters"] > 0 for c in data["controls"])
    assert data["novelty_preflight"]["status"] == "passed"
    assert all("sha256_forward" in c["audit"] for c in data["controls"])

def test_audit_is_exactly_letter_level():
    assert audit("A man, a plan, a canal: Panama!")["two_pointer_exact"]
    assert not audit("A careful gardener covered the seedlings.")["two_pointer_exact"]
