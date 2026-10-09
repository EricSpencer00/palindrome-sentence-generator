"""Small exact-search fixtures; their strings are not readability evidence."""
import pytest

from llm_palindrome.search import WordTries, beam_search, unit_letters


class IncreasingScorer:
    def word_delta(self, left, right, placement, word, growth):
        return float(len(left + right) ** 2)


def test_long_phrase_debt_can_close_when_explicitly_allowed():
    phrase = "one two three four five six seven"
    letters = unit_letters(phrase)
    assert len(letters) > 24
    tries = WordTries([phrase, letters[::-1]])
    options = dict(min_letters=2 * len(letters), max_steps=3, beam_width=4,
                   diversity=0.0, max_word_uses=1)
    assert beam_search(tries, IncreasingScorer(), **options) == []
    for cap in (96, None):
        result = beam_search(tries, IncreasingScorer(), max_overhang=cap, **options)
        norm = "".join(map(unit_letters, result))
        assert len(norm) == 2 * len(letters)
        assert norm == norm[::-1]
        assert phrase in result


def test_disabling_incumbent_stop_reaches_later_better_distinct_closure():
    # One exact path isolates stopping behavior from stochastic beam ranking.
    tries = WordTries(["a"])
    options = dict(min_letters=1, max_steps=8, beam_width=1, diversity=0.0)
    historical = beam_search(tries, IncreasingScorer(), **options)
    explicit = beam_search(tries, IncreasingScorer(), closed_stop_factor=3.0, **options)
    continued = beam_search(tries, IncreasingScorer(), closed_stop_factor=None, **options)
    assert historical == explicit == ["a"] * 3
    assert continued == ["a"] * 7
    objective = lambda words: sum(i * i for i in range(1, len(words) + 1)) / len(words)
    assert objective(continued) > objective(historical)
    assert "".join(continued) == "".join(continued)[::-1]


@pytest.mark.parametrize("parameter", ["max_overhang", "closed_stop_factor"])
@pytest.mark.parametrize("value", [0, -1, float("nan")])
def test_nonpositive_or_nan_caps_are_rejected(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        beam_search(WordTries(["a"]), IncreasingScorer(), **{parameter: value})
