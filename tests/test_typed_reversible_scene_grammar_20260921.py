from typed_reversible_scene_grammar_20260921 import pointer_audit, run

def test_independent_pointer_and_sha_validation():
    audit = pointer_audit("Never odd or even")
    assert audit["pointer_exact"]
    assert audit["sha256_forward"] == audit["sha256_reverse"]

def test_signature_bucket_controls_and_provenance():
    result = run()
    assert result["stats"]["rendered_controls"] == 4
    assert result["novelty_preflight"]["status"] == "passed"
    for row in result["rendered_candidates"]:
        assert row["provenance"]["independent_side_generation"]
        assert row["gates"]["grammar_slots_complete"]
        assert row["gates"]["reader_verified_readability"] is False
