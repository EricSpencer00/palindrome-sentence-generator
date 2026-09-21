from experiments.abba_centerout_semantic_seam_20260922 import run, audit, center_product

def test_independent_audit_and_center_trace():
    d=run(); assert d["stats"]["semantic_states"] == 3
    assert d["stats"]["exact_gt38"] == 0
    for row in d["rendered_candidates"]:
        assert row["audit"]["letters"] > 38
        assert row["provenance"]["center_out_online"]
        assert row["provenance"]["finished_text_reversal"] is False
        assert row["center_product"]["paired_checks"] > 0

def test_center_product_detects_mismatch_independently():
    p=center_product("ab", "ac")
    assert p["first_mismatch"]["depth"] == 0
    assert audit("Able was I ere I saw Elba.")["two_pointer_exact"]
