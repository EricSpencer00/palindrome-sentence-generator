from experiments.typed_determiner_adjunct_seam_20260918 import run, discover, _independent_audit
from experiments.dream_rsi_exact_boundary_20260918 import audit

def test_typed_prep_lane_has_independent_audits():
    p = run()
    assert p["construction"]["preposition_valency_state"]
    assert p["construction"]["animate_inanimate_object_typing"]
    for c in p["rendered_candidates"]:
        assert c["audit"] == _independent_audit(c["rendered"])
        assert c["reference_audit"] == audit(c["rendered"])

def test_typed_equations_prune_before_rendering():
    p = discover("short_first", 50)
    assert p["dead_frontier"]
    assert all("equation" in n for n in p["nodes"])
