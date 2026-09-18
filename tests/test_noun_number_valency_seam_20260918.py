from experiments.noun_number_valency_seam_20260918 import audit, discover, run

def test_valency_lane_has_live_equations_and_independent_audit():
    payload = run()
    assert payload["construction"]["noun_number_carried"]
    assert payload["construction"]["semantic_valency_checked"]
    assert payload["novelty_preflight"]["prior_lane_reused"] is False
    for row in payload["rendered_candidates"]:
        assert row["audit"] == audit(row["rendered"])
        assert row["provenance"]["catalogue_used"] is False

def test_valency_search_prunes_before_rendering():
    report = discover("short_first", budget=30)
    assert report["stats"]["nodes"] <= 30
    assert all("rendered" not in n for n in report["nodes"])
