from experiments.clause_product_transitive_intransitive_20260920 import audit, coupled, run

def test_online_coupling_crosses_word_boundaries():
    assert coupled("a quiet", "teiuq a")[0] >= 1
    assert coupled("a quiet", "zz")[1] is False

def test_run_has_independent_semantic_product_and_audits():
    out = run()
    assert out["stats"]["product_states"] == 100
    assert out["novelty_preflight"]["finished_tape_reversal"] is False
    assert out["novelty_preflight"]["post_hoc_repair"] is False
    assert all("sha256_forward" in r["audit"] for r in out["near_misses"])

def test_audit_two_pointer_and_sha_are_agreeing():
    row = audit("level")
    assert row["exact"] and row["sha_equal"]
