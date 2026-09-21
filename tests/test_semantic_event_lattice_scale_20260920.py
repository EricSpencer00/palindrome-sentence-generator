from experiments.semantic_event_lattice_scale_20260920 import audit, run

def test_scaled_lattice_prunes_semantics_before_characters_and_keeps_controls():
    r=run(); s=r["stats"]
    assert s["event_pairs_expanded"] == r["config"]["events"] ** 2
    assert s["semantic_valid_edges"] > 0 and s["semantic_prunes"] > 0
    assert s["obligation_states"] > s["semantic_valid_edges"]
    assert s["prose_controls"] > 0 and s["exact_gt38"] == 0
    assert all(not x["reader_eligible"] for x in r["prose_controls"])
    assert r["provenance"]["independent_audit"] == ["two-pointer", "forward/reverse SHA-256"]

def test_audit_is_independent():
    x=audit("The guard opened the gate, then the keeper lifted the flag.")
    assert not x["pointer_exact"] and x["sha256_forward"] != x["sha256_reverse"]
