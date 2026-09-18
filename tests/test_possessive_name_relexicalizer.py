import pytest

from experiments.possessive_name_relexicalizer import audit, render, run
from experiments.verify_possessive_name_candidates import (
    hard_gates_independently,
    normalize_independently,
    source_contract_errors,
)


def test_exact_catalogue_family_variant_is_rejected_even_when_its_letters_close():
    text = render("marge", "aino", "sonia", "telegram")
    checks = audit(text, local_catalogue=set())
    assert text == "Marge lets Aino see Sonia's telegram."
    assert checks["exact_letter_palindrome"] is True
    assert checks["not_catalogue_family_derivative"] is False
    assert normalize_independently(text) == normalize_independently(text)[::-1]


def test_catalogue_control_is_explicitly_excluded_not_reframed_as_generated():
    text = render("marge", "norah", "sharon", "telegram")
    checks = audit(text, local_catalogue=set())
    assert checks["exact_letter_palindrome"] is True
    assert checks["not_forbidden_catalogue_control"] is False
    assert checks["not_catalogue_family_derivative"] is False


def test_independent_auditor_rejects_a_catalogue_family_with_palindromic_wrappers():
    checks = hard_gates_independently(
        "Part Marge lets Norah see Sharon's telegram trap.", set()
    )
    assert checks["exact_letter_palindrome"] is True
    assert checks["not_catalogue_family_derivative"] is False


def test_run_preserves_exact_closures_but_promotes_no_catalogue_family_variant():
    result = run()
    rendered = {row["rendered"] for row in result["mechanically_admitted"]}
    assert rendered == set()
    assert all(row["checks"]["exact_letter_palindrome"] for row in result["exact_closures"])


def test_independent_auditor_recomputes_every_gate_without_source_check_trust():
    text = render("marge", "aino", "sonia", "telegram")
    checks = hard_gates_independently(text, set())
    assert checks["exact_letter_palindrome"] is True
    assert checks["not_catalogue_family_derivative"] is False


def test_wordwise_reversal_is_rejected_even_when_opposite_words_differ():
    checks = audit("Step on no pets.", local_catalogue=set())
    independent = hard_gates_independently("Step on no pets.", set())
    assert checks["exact_letter_palindrome"] is True
    assert checks["not_word_order_symmetry"] is False
    assert independent["not_word_order_symmetry"] is False


def test_non_ascii_alphabetic_input_is_rejected_in_both_normalizers():
    checks = audit("Márge lets Hara see Sarah's telegram.", local_catalogue=set())
    assert checks["supported_ascii_letters"] is False
    with pytest.raises(ValueError, match="non-ASCII"):
        normalize_independently("Márge lets Hara see Sarah's telegram.")


def test_malformed_source_is_rejected_before_independent_enumeration():
    assert "source vocabulary is malformed" in source_contract_errors({"vocabulary": None})


def test_source_contract_rejects_mutated_frozen_vocabulary_or_closures():
    vocabulary_mutation = run()
    vocabulary_mutation["vocabulary"]["subjects"] = ("eve",)
    assert "source vocabulary differs from the frozen ablation" in source_contract_errors(vocabulary_mutation)

    closure_mutation = run()
    closure_mutation["exact_closures"] = []
    assert "source exact closures differ from independent enumeration" in source_contract_errors(closure_mutation)
