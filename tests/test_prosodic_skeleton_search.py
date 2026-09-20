from prosodic_skeleton_search_20260920 import audit, run

def test_audit_is_independent_exact_gate():
    assert audit("A man, a plan, a canal: Panama!")["exact"]
    assert not audit("the quiet teacher marks a new route")["exact"]

def test_fresh_lane_has_no_shortcut_and_artifact_stats():
    result = run()
    assert result["novelty_preflight"]["catalogue_surface_reuse"] is False
    assert result["novelty_preflight"]["live_construction_equation"] is False
    assert result["novelty_preflight"]["palindrome_constructor"] is False
    assert result["status"] == "diagnostic prose controls; no-reader claim; no palindrome constructor"
    assert result["stats"]["rendered_candidates"] == 16
    assert all(not row["provenance"]["finished_tape_reversal"] for row in result["rendered_candidates"])
