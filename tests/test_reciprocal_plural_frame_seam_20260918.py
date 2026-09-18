from experiments.reciprocal_plural_frame_seam_20260918 import run, discover
from experiments.dream_rsi_exact_boundary_20260918 import audit

def test_reciprocal_lane_has_live_equations_and_independent_audit():
    p = run()
    assert p["construction"]["reciprocal_lexicon"]
    assert p["construction"]["live_character_equations"]
    assert p["stats"]["fresh_nodes"] >= 0
    for c in p["rendered_candidates"]:
        assert c["audit"] == audit(c["rendered"])

def test_discover_prunes_before_rendering():
    p = discover("short_first", 50)
    assert p["dead_frontier"]
    assert all("equation" in n for n in p["nodes"])
