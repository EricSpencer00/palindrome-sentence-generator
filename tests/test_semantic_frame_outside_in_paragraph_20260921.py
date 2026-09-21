from experiments.semantic_frame_outside_in_paragraph_20260921 import exact_audit, generate, run

def test_audit_is_independent_pointer_check():
    assert exact_audit("Never odd or even") ["pointer_exact"]
    assert not exact_audit("A quiet harbor") ["pointer_exact"]

def test_rendered_candidate_has_semantic_plan_and_live_decisions():
    c = generate(6)
    assert len(c["semantic_plan"]) == 6
    assert len(c["residual_decisions"]) == 6
    assert c["gates"]["no_post_hoc_repair"]
    assert c["provenance"]["repair_passes"] == 0

def test_scalable_lengths_and_novelty_preflight():
    result = run()
    assert result["stats"]["max_letters"] > 100
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["candidates"] == 3
