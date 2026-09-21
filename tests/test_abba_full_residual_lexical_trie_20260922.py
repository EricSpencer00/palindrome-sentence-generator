from experiments.abba_full_residual_lexical_trie_20260922 import audit, build_trie, run

def test_audit_is_independent():
    assert audit("An aide rips nine memos; some men inspire Diana.")["two_pointer_exact"]
    assert not audit("The careful gardener waters the garden.")["two_pointer_exact"]

def test_full_residual_trie_lane_has_fresh_controls_and_certificate():
    data = run()
    assert data["stats"]["branches"] == 4
    assert data["stats"]["closed_derivations"] == 0
    assert data["stats"]["exact_gt38"] == 0
    assert len(data["controls"]) == 4
    assert all(c["audit"]["two_pointer_exact"] is False for c in data["controls"])
    assert all(b["full_residual_prefix"] for b in data["residual_certificate"])

def test_trie_contains_character_edges():
    trie = build_trie({"subject": ("a quiet sailor",), "verb": ("studies",),
                       "object": ("a folded map",), "adjunct": ("before dawn",)})
    assert "a" in trie["children"]
