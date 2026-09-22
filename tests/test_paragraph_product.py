from llm_palindrome.paragraph_product import (
    audit_staggered_abba,
    staggered_abba_search,
)


def test_staggered_boundary_audit_distinguishes_cross_sentence_exactness():
    audit = audit_staggered_abba(("Abcd.", "Efghij."),
                                 ("Jih.", "Gfedcba."))
    assert audit["two_pointer_exact"] is True
    assert audit["left_boundaries"] == [4]
    assert audit["reflected_right_boundaries"] == [7]
    assert audit["sentence_boundaries_staggered"] is True
    assert audit["whole_sentence_mirrors"] == []
    assert audit["cross_sentence_coupled"] is True


def test_aligned_sentence_mirror_pairs_are_rejected_as_preclosed_blocks():
    audit = audit_staggered_abba(("Abcd.", "Efghij."),
                                 ("Jihgfe.", "Dcba."))
    assert audit["two_pointer_exact"] is True
    assert audit["aligned_internal_boundaries"] == [4]
    assert audit["whole_sentence_mirrors"]
    assert audit["cross_sentence_coupled"] is False


def test_search_keeps_only_independently_staggered_sentence_parses():
    left = (
        (("A", ("abcd",)),),
        (("B", ("efghij",)),),
    )
    right = (
        (("B-prime", ("jih",)),),
        (("A-prime", ("gfedcba",)),),
    )
    result = staggered_abba_search(left, right)
    assert result["cap_reached"] is False
    assert len(result["cross_sentence_candidates"]) == 1
    row = result["cross_sentence_candidates"][0]
    assert row["rendered"] == "Abcd. Efghij. Jih. Gfedcba."
    assert row["audit"]["half_equation"] is True
