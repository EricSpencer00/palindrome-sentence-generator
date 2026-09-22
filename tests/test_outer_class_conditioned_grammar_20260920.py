from experiments.outer_class_conditioned_grammar_20260920 import audit, run

def test_outer_class_lane_is_forward_and_audited():
    r=run()
    assert r["novelty_preflight"]["status"]=="passed"
    assert r["stats"]["rendered_candidates"] > 0
    assert r["stats"]["fresh_exact_gt38"] == 0
    for row in r["rendered_candidates"]:
        assert row["provenance"]["endpoint_indexed_before_interior"]
        assert not row["provenance"]["finished_tape_reversal"]
        assert len(row["audit"]["sha256_forward"]) == 64

def test_independent_letter_audit():
    assert audit("A man, a plan, a canal: Panama!")["exact"]
