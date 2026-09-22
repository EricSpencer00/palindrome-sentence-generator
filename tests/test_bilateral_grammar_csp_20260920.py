from itertools import product
import pytest
from experiments.bilateral_grammar_csp_20260920 import bilateral_grammar_csp, _consume, palindromic_residual
from experiments.forward_lexicalized_grammar_20260920 import ATOMIC, Word, letters


ANCHOR_LEXICON = tuple(
    word for word in ATOMIC
    if word.text in {"an", "aide", "rips", "nine", "memos", "some", "men", "inspire", "diana"}
)


def test_bilateral_grammar_recovers_anchor_without_injection():
    result = bilateral_grammar_csp(ANCHOR_LEXICON, max_nodes=100_000)
    rows = [row for row in result["paths"] if letters(row["rendered"]) ==
            letters("An aide rips nine memos; some men inspire Diana.")]
    assert rows
    assert rows[0]["audit"]["exact"]
    assert rows[0]["provenance"]["right_clause_reverse_expansion"]


def test_bilateral_grammar_rejects_shortcut_inventory():
    result = bilateral_grammar_csp(
        tuple(word for word in ATOMIC if word.text in {"madam", "redivider", "a", "the", "dog"}),
        max_nodes=10_000,
    )
    assert not result["paths"]


def test_center_rule_matches_exhaustive_independent_oracle():
    words = ["".join(chars) for n in range(1, 5) for chars in product("ab", repeat=n)]
    old_count = new_count = 0
    for left in words:
        for right in words:
            tape = left + right
            expected = all(tape[i] == tape[-1-i] for i in range(len(tape) // 2))
            residual = _consume(left, right[::-1])
            actual = residual is not None and palindromic_residual(*residual)
            assert actual == expected, (left, right)
            old_count += residual == ("", "")
            new_count += actual
    assert (old_count, new_count) == (30, 126)


@pytest.mark.parametrize("left,right,accepted", [
    ("ab", "cba", True), ("a", "bba", True), ("ab", "ccba", True), ("abc", "ba", True),
    ("ab", "dcba", False), ("ab", "da", False),
])
def test_actual_solver_closes_centers_inside_terminal_words(left, right, accepted):
    result = bilateral_grammar_csp(
        (Word(left, "L"), Word(right, "R")), grammar={"UNUSED": ()},
        left_symbols=("L",), right_symbols=("R",), max_words=2, max_nodes=20,
    )
    assert bool(result["paths"]) == accepted
    if accepted:
        row = result["paths"][0]
        assert row["audit"]["exact"]
        assert row["audit"]["sha256_forward"] == row["audit"]["sha256_reverse"]
        assert row["center_residual"]
