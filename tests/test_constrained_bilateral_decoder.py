from experiments.constrained_bilateral_decoder import (
    _allow_closed, _allow_state, observed_joins, split_bilateral,
)
from llm_palindrome.bigram import BigramModel


def test_split_recovers_distinct_word_boundaries_of_an_exact_tape():
    assert split_bilateral(["step", "on", "a", "a", "no", "pets"]) == (
        "step on a", "a no pets")


def test_split_rejects_a_nonmirrored_word_sequence():
    assert split_bilateral(["this", "is", "not", "a", "palindrome", "today"]) is None


def test_decoder_state_enforces_no_repeated_words_and_closed_side_bands():
    assert _allow_state(("step", "on"), ("no", "pets"))
    assert not _allow_state(("step",), ("on", "step"))
    assert _allow_closed(("they", "saw", "three"), ("we", "found", "six"))
    assert not _allow_closed(("one", "two"), ("three", "four", "five"))


def test_decoder_rejects_fragment_openings_and_trailing_function_words():
    assert not _allow_closed(("met", "in", "position"), ("no", "it", "item"))
    assert not _allow_closed(("they", "found", "on"), ("we", "saw", "item"))


def test_observed_joins_is_a_filter_not_a_language_quality_claim():
    model = BigramModel({("a", "cat"): 2}, {"a": 2, "cat": 2})
    assert observed_joins("a cat", model)
    assert not observed_joins("cat a", model)


def test_state_can_require_attested_joins_while_expanding():
    model = BigramModel({("a", "cat"): 2, ("no", "pets"): 2},
                        {"a": 2, "cat": 2, "no": 2, "pets": 2})
    assert _allow_state(("a", "cat"), ("no", "pets"), bigrams=model)
    assert not _allow_state(("cat", "a"), ("no", "pets"), bigrams=model)
