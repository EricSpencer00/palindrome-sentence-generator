from experiments.memoized_residual_palindrome import construct
from llm_palindrome.validator import normalize, is_palindrome

def test_constructs_exact_requested_length():
    row = construct(47)
    assert row["status"] == "exact"
    assert row["letters"] == 47
    assert is_palindrome(row["text"])
    assert row["independent_validation"]

def test_reports_unreachable_residual_without_faking_output():
    row = construct(39)
    assert row["status"] == "no_construction"
    assert "text" not in row
