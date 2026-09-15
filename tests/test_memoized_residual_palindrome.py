from experiments.memoized_residual_palindrome import construct
from llm_palindrome.validator import normalize, is_palindrome

def test_constructs_exact_requested_length_with_boundary_crossing_units():
    row = construct(30)
    assert row["status"] == "exact"
    assert row["letters"] == 30
    assert is_palindrome(row["text"])
    assert row["independent_validation"]
    assert row["admission"]["admitted"]

def test_marks_old_semordnilap_chain_inadmissible():
    row = construct(47)
    assert row["status"] == "no_construction"
