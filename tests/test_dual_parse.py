from llm_palindrome.admission import mechanical_admission_checks
from llm_palindrome.dual_parse import SurfaceLattice, intersect_surfaces, letter_tape


SEED = "An aide rips nine memos; some men inspire Diana."


def test_dual_parse_recovers_seed_with_different_roles_and_boundaries():
    left = SurfaceLattice()
    left.slot("A:agent", ["An aide"])
    left.slot("B:event", ["rips nine memos;"])

    right = SurfaceLattice()
    right.slot("B-prime:response", ["some men inspire"])
    right.slot("A-prime:patient", ["Diana."])

    search = intersect_surfaces(left, right)
    assert search["cap_reached"] is False
    assert len(search["results"]) == 1
    result = search["results"][0]
    assert result["rendered"] == SEED
    assert result["exact_half_equation"] is True
    assert result["different_word_segmentation"] is True
    assert result["left_roles"] == ["A:agent", "B:event"]
    assert result["right_roles"] == ["B-prime:response", "A-prime:patient"]
    assert letter_tape(result["rendered"]) == letter_tape(result["rendered"])[::-1]
    assert all(mechanical_admission_checks(result["rendered"], max_letters=100).values())


def test_dual_parse_prunes_incompatible_english_surfaces_before_rendering():
    left = SurfaceLattice()
    left.slot("A", ["The careful keeper"])
    left.slot("B", ["opens the gate."])
    right = SurfaceLattice()
    right.slot("B-prime", ["A patient sailor"])
    right.slot("A-prime", ["returns at dusk."])

    search = intersect_surfaces(left, right)
    assert search["results"] == []
    assert search["states"] == 1
    assert search["transitions"] == 0
    assert search["dead_frontiers"][0]["left_next"] == ["t"]
    assert search["dead_frontiers"][0]["right_next"] == ["k"]
