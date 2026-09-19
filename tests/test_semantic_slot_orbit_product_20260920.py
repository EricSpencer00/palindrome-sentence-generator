from experiments import semantic_slot_orbit_product_20260920 as lane

def test_preflight_and_independent_audits():
    result = lane.run(max_states=20)
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["states"] == 20
    assert result["stats"]["exact"] == 0
    assert result["provenance"]["repair_after_render"] is False
    assert result["provenance"]["agreement_carrying_morphology"]
    for row in result["controls"]:
        assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]
        assert row["semantic_slots"]["left_valency"] == "transitive"
        assert row["orbit_equations"]

def test_frames_have_attachment_and_agreement_variants():
    assert len(lane.FRAMES) == 4
    assert any(f.verb_sg != f.verb_pl for f in lane.FRAMES)
    assert all(f.attach and f.prep for f in lane.FRAMES)
