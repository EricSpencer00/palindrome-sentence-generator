from experiments.semantic_valency_dual_parse_20260921 import run


def test_semantic_valency_product_recovers_seed_but_promotes_nothing_short():
    result = run()
    assert result["stats"]["recovery_controls"] == 1
    assert result["stats"]["fresh_mechanically_admitted_gt38"] == 0
    assert result["fresh_exact_candidates"] == []
    control = result["recovery_controls"][0]
    assert control["audit"]["two_pointer_exact"] is True
    assert control["mechanical_admission"]["length_band"] is False


def test_semantic_valency_product_carries_roles_before_exact_closure():
    result = run()
    assert result["stats"]["states"] > 0
    assert result["deepest_frontiers"]
    assert all("next_left_role" in row and "next_right_role" in row
               for row in result["deepest_frontiers"])
    assert result["provenance"]["per_candidate_rlaif"] is False
