from bilateral_grammar_csp_20260920 import bilateral_grammar_csp
from forward_lexicalized_grammar_20260920 import ATOMIC, letters


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
