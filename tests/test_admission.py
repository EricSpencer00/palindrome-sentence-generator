import pytest

from llm_palindrome.admission import (
    has_distinct_content_words,
    has_forbidden_catalogue_endpoint_scaffold,
    has_only_ordinary_short_words,
    phrasewise_reverse_boundary_offsets,
    has_repeated_nontrivial_unit,
    has_self_palindromic_proper_multiword_span,
    mechanical_admission_checks,
    normalize_letters,
)


def test_catalogue_relexicalization_is_fail_closed_even_with_changed_names():
    checks = mechanical_admission_checks("Marge lets Hara see Sarah's telegram.")
    assert checks["exact_letter_palindrome"] is True
    assert checks["not_catalogue_family_derivative"] is False

    punctuation_free = mechanical_admission_checks(
        "Marge lets Hara see Sarahs telegram.", max_letters=80
    )
    assert punctuation_free["not_catalogue_family_derivative"] is False

    wrapped = mechanical_admission_checks(
        "Part Marge lets Norah see Sharon's telegram trap.", max_letters=100
    )
    assert wrapped["exact_letter_palindrome"] is True
    assert wrapped["not_catalogue_family_derivative"] is False


def test_nonwords_cannot_be_mechanically_admitted() -> None:
    checks = mechanical_admission_checks(
        "Abcde fghijk lmnop ponml kjihg fedcba.", max_letters=100
    )
    assert checks["exact_letter_palindrome"] is True
    assert checks["lexicon_words"] is False


def test_exact_gibberish_cannot_be_labeled_proper_names_to_bypass_the_gate():
    # This is a letter-level palindrome.  A prior gate accepted it when a
    # candidate author passed each invented token as an "allowed" name.
    checks = mechanical_admission_checks(
        "Abc def ghij klm nop qrst tsr qponm lkj ihg fed cba.",
        min_letters=30,
        max_letters=80,
    )
    assert checks["exact_letter_palindrome"] is True
    assert checks["lexicon_words"] is False


def test_dictionary_fragments_fail_the_shared_short_word_filter():
    assert has_only_ordinary_short_words(("no", "it", "can", "act"))
    assert not has_only_ordinary_short_words(("draw", "la", "tips", "st"))


def test_reverse_whole_word_construction_is_not_admitted():
    checks = mechanical_admission_checks(
        "Rats stressed deliver diaper repaid reviled desserts star."
    )
    assert checks["exact_letter_palindrome"] is True
    assert checks["not_word_order_symmetry"] is False


def test_phrasewise_reverse_boundary_gate_rejects_segmented_local_equation():
    left = "Reda Mar diapered Elena."
    right = "Ane Lede repaid Rama Der."
    left_tape = normalize_letters(left)
    right_tape = normalize_letters(right)
    assert left_tape == right_tape[::-1]
    assert phrasewise_reverse_boundary_offsets(left, right) == (7,)


def test_phrasewise_reverse_boundary_gate_preserves_the_38_letter_seed():
    # Split the classic seed at its central word gap. The two intact clauses
    # close by letters, but have no shared reflected internal word seam.
    left = "An aide rips nine memos"
    right = "some men inspire Diana"
    assert normalize_letters(left) == normalize_letters(right)[::-1]
    assert phrasewise_reverse_boundary_offsets(left, right) == ()


def test_phrasewise_reverse_boundary_gate_only_reports_exact_equations():
    assert phrasewise_reverse_boundary_offsets("A baker cools one tart", "Some diners share data") == ()


def test_paragraph_line_breaks_are_valid_rendering_but_not_a_gate_bypass():
    text = "An aide rips nine memos;\n\nsome men inspire Diana."
    checks = mechanical_admission_checks(text, max_letters=100)
    assert checks["word_form"] is True
    assert all(checks.values())


def test_unsupported_unicode_fails_closed_instead_of_silently_disappearing():
    checks = mechanical_admission_checks("Márge lets Hara see Sarah's telegram.")
    assert checks["supported_ascii_letters"] is False
    with pytest.raises(ValueError, match="non-ASCII"):
        normalize_letters("Márge")


def test_separated_repeated_multiword_unit_is_rejected():
    assert has_repeated_nontrivial_unit(("alpha", "beta", "gamma", "alpha", "beta"))
    assert not has_repeated_nontrivial_unit(("that", "a", "careful", "artist", "that", "a", "brave", "guard"))


def test_hidden_self_palindromic_proper_multiword_span_is_rejected():
    units = ("careful", "to", "order", "red", "root", "artists")
    assert has_self_palindromic_proper_multiword_span(units)
    checks = mechanical_admission_checks("Careful to order red root artists.")
    assert checks["no_self_palindromic_proper_multiword_span"] is False

    # The full rendered candidate is necessarily a palindrome; only a proper
    # embedded span is a prohibited prebuilt construction unit.
    assert not has_self_palindromic_proper_multiword_span(("to", "order", "red", "root"))


def test_articles_and_complementizers_may_repeat_but_content_words_may_not():
    assert has_distinct_content_words((
        "a", "quiet", "portrait", "that", "a", "careful", "artist", "praised",
    ))
    assert not has_distinct_content_words(("a", "quiet", "portrait", "that", "an", "artist", "praised", "portrait"))
    checks = mechanical_admission_checks("A quiet artist that a brave guard greeted.")
    assert checks["no_self_palindromic_word"]


def test_rendered_catalogue_entry_is_normalized_before_membership_check():
    checks = mechanical_admission_checks(
        "Satan, oscillate my metallic sonatas!",
        local_catalogue=["satan oscillate my metallic sonatas"],
    )
    assert checks["exact_letter_palindrome"] is True
    assert checks["absent_from_local_catalogue"] is False


def test_repository_catalogue_is_a_central_nonoptional_exclusion():
    # This test deliberately supplies no caller catalogue. A constructor must
    # not be able to admit a borrowed classic by simply omitting that argument.
    checks = mechanical_admission_checks("Satan, oscillate my metallic sonatas!")
    assert checks["exact_letter_palindrome"] is True
    assert checks["absent_from_local_catalogue"] is False


def test_newly_discovered_catalogue_item_is_centrally_excluded():
    # This exact surface appears in an external palindrome catalogue.  It must
    # remain ineligible even when a constructor supplies no local list.
    checks = mechanical_admission_checks("Race not one car.")
    assert checks["exact_letter_palindrome"] is True
    assert checks["absent_from_local_catalogue"] is False

    checks = mechanical_admission_checks("Name not one man.")
    assert checks["exact_letter_palindrome"] is True
    assert checks["absent_from_local_catalogue"] is False


def test_catalogue_endpoint_scaffolds_are_rejected_even_with_a_new_middle():
    # These are not claimed to be palindromes.  That is the point: an outer
    # classic cannot become eligible merely by inserting fresh material.
    assert has_forbidden_catalogue_endpoint_scaffold(("no", "it", "is", "fresh", "position"))
    assert has_forbidden_catalogue_endpoint_scaffold(("see", "the", "garden", "bees"))
    assert has_forbidden_catalogue_endpoint_scaffold(("go", "finish", "the", "dog"))
    assert not has_forbidden_catalogue_endpoint_scaffold(("can", "writers", "serve", "cognac"))

    checks = mechanical_admission_checks("See the careful keeper feed hungry bees.")
    assert checks["not_forbidden_catalogue_endpoint_scaffold"] is False
