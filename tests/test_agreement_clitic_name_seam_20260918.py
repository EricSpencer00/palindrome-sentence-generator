from experiments.agreement_clitic_name_seam_20260918 import audit, discover, run


def test_seam_lane_has_live_equations_and_independent_audit():
    payload = run()
    assert payload["construction"]["live_character_equations"]
    assert payload["construction"]["name_adjacent_seams"]
    assert payload["novelty_preflight"]["prior_lane_reused"] is False
    for row in payload["rendered_candidates"]:
        assert row["audit"] == audit(row["rendered"])
        assert row["provenance"]["catalogue_used"] is False


def test_discover_prunes_before_rendering():
    report = discover("short_first", budget=100)
    assert report["stats"]["nodes"] <= 100
    assert all("rendered" not in node for node in report["nodes"])
