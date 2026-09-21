from experiments.abba_relation_first_trie_20260922 import audit, run

def test_seed_audit():
    assert audit("An aide rips nine memos; some men inspire Diana.")["two_pointer_exact"]

def test_relation_first_lane_has_controls_and_residual_certificate():
    d=run()
    assert d["stats"]["semantic_states"] == 3
    assert d["stats"]["closed_derivations"] == 0
    assert d["stats"]["exact_gt38"] == 0
    assert len(d["controls"]) == 12
    assert all(not x["audit"]["two_pointer_exact"] for x in d["controls"])
    assert all(x["residual_prefix"] for x in d["residual_certificates"])
    assert d["novelty_preflight"]["status"] == "passed"
