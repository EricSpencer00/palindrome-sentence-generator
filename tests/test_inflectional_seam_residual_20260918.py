from experiments.inflectional_seam_residual_20260918 import run

def test_lane_has_cross_word_provenance_and_independent_audit():
    r = run()
    assert r["novelty_preflight"]["duplicate_sweep"] is False
    assert r["stats"]["index_entries"] > 0
    for x in r["rendered_candidates"]:
        assert x["provenance"]["complete_grammatical_clauses"]
        assert x["seam"]["crosses_word_boundary"]
        assert set(x["audit"]) >= {"two_pointer_exact", "sha256_forward", "sha256_reverse"}
