from llm_palindrome.admission import mechanical_admission_checks
from llm_palindrome.dual_parse import (
    Morphology,
    SurfaceLattice,
    intersect_surfaces,
    letter_tape,
    productive_lattice,
    word_residual_search,
)


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


def test_word_residual_search_recovers_seed_in_reading_order():
    left = (
        ("A:determiner", ("an",)),
        ("A:agent", ("aide",)),
        ("B:verb", ("rips",)),
        ("B:quantity", ("nine",)),
        ("B:object", ("memos",)),
    )
    right = (
        ("B-prime:response", ("some",)),
        ("B-prime:agent", ("men",)),
        ("B-prime:verb", ("inspire",)),
        ("A-prime:patient", ("Diana",)),
    )
    search = word_residual_search(left, right)
    assert [row["rendered"] for row in search["results"]] == [
        "an aide rips nine memos some men inspire Diana"
    ]
    assert search["results"][0]["exact_half_equation"] is True


def test_word_residual_search_reports_the_deepest_failed_obligation():
    search = word_residual_search(
        (("left-determiner", ("an",)), ("left-agent", ("elder",))),
        (("right-event", ("returns",)), ("right-name", ("Lena",))),
    )
    assert search["results"] == []
    assert search["dead_frontiers"]
    deepest = search["dead_frontiers"][0]
    assert deepest["matched_letters"] >= 2
    assert deepest["owner"] in {"left", "right"}
    assert deepest["residual"]


def test_productive_lattice_attaches_features_to_repeated_role_slots():
    first = Morphology(noun_number="singular")
    second = Morphology(noun_number="plural")
    lattice = productive_lattice([
        ("noun", [("pilot", first)]),
        ("noun", [("sailors", second)]),
    ])
    assert [lattice.morphology[index] for index in sorted(lattice.morphology)] == [
        first,
        second,
    ]


def test_word_residual_search_matches_brute_force_on_tiny_grammars():
    from itertools import product

    left = (("left-1", ("a", "ab")), ("left-2", ("c", "bc")))
    right = (("right-1", ("a", "ca")), ("right-2", ("c", "cba")))
    expected = {
        (" ".join(left_words), " ".join(right_words))
        for left_words in product(*(slot[1] for slot in left))
        for right_words in product(*(slot[1] for slot in right))
        if letter_tape(" ".join(left_words))
        == letter_tape(" ".join(right_words))[::-1]
    }
    search = word_residual_search(left, right, max_states=10_000)
    actual = {(row["left"], row["right"]) for row in search["results"]}
    assert actual == expected
    assert search["cap_reached"] is False


def test_word_residual_partial_gate_prunes_before_closure():
    left = (("left-1", ("a",)), ("left-2", ("a",)))
    right = (("right-1", ("a",)), ("right-2", ("a",)))
    unrestricted = word_residual_search(left, right)
    gated = word_residual_search(
        left,
        right,
        allow_partial=lambda left_words, right_words:
            len(left_words + right_words) == len(set(left_words + right_words)),
    )
    assert unrestricted["results"]
    assert gated["results"] == []
    assert gated["states"] < unrestricted["states"]


def test_word_residual_can_reject_composed_interior_closures():
    left = (("left-a", ("a",)), ("left-b", ("b",)))
    right = (("right-b", ("b",)), ("right-a", ("a",)))
    composed = word_residual_search(left, right)
    irreducible = word_residual_search(
        left,
        right,
        reject_intermediate_closure=True,
    )
    assert composed["results"]
    assert irreducible["results"] == []
    assert irreducible["intermediate_closure_rejections"] > 0
