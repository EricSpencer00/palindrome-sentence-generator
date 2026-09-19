from experiments.finite_clause_character_orbits_20260919 import audit, run


def test_live_orbit_lane_requires_complete_svo_terminals_and_keeps_controls():
    result = run()

    assert result["stats"]["exact"] == 0
    assert result["stats"]["exact_closures_before_novelty"] == 0
    assert result["stats"]["expanded_states"] == 8
    assert result["stats"]["intact_controls"] == 3
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["novelty_preflight"]["reversible_lexical_pairs_used"] is False
    assert result["novelty_preflight"]["finished_tape_reversed"] is False
    assert result["novelty_preflight"]["word_order_symmetry_used"] is False
    assert result["novelty_preflight"]["rlaif_used"] is False

    controls = result["rendered_candidates"]
    assert all(row["source"] == "intact-finite-svo-control" for row in controls)
    assert all(row["complete_finite_svo"] for row in controls)
    assert all(39 <= row["audit"]["letters"] <= 60 for row in controls)
    assert all(not row["audit"]["two_pointer_exact"] for row in controls)
    assert all(not row["mechanically_admitted"] for row in controls)
    assert all(row["orbit_assignment"]["assigned_orbits"] > 0 for row in controls)

    first = controls[0]
    assert first["rendered"] == "An aide writes nine memos; Some men inspire Diana."
    assert first["audit"]["sha256_forward"] != first["audit"]["sha256_reverse"]
    assert first["orbit_assignment"]["first_failure"]["orbit"] == 6
    assert audit(first["rendered"])["two_pointer_exact"] is False


def test_orbit_lane_does_not_promote_a_partial_clause():
    result = run()
    assert all(row["roles"]["left"]["subject"] for row in result["rendered_candidates"])
    assert all(row["roles"]["left"]["finite_verb"] for row in result["rendered_candidates"])
    assert all(row["roles"]["left"]["object"] for row in result["rendered_candidates"])
    assert all(row["roles"]["right"]["subject"] for row in result["rendered_candidates"])
    assert all(row["roles"]["right"]["finite_verb"] for row in result["rendered_candidates"])
    assert all(row["roles"]["right"]["object"] for row in result["rendered_candidates"])
