from llm_palindrome.search import State, WordTries, beam_search
from llm_palindrome.centerout import centerout_search


class ZeroScorer:
    def word_delta(self, left, right, placement, word, growth):
        return 0.0


def test_beam_search_can_continue_from_exact_compatible_endpoint_state():
    state = State(0.0, ("some",), ("memos",), "m", "L", 0.0)
    result = beam_search(
        WordTries(["some", "memos", "a", "am", "man"]),
        ZeroScorer(),
        min_letters=9,
        beam_width=8,
        max_steps=2,
        candidate_limit=32,
        initial_state=state,
    )
    assert result[:1] == ["some"]
    assert result[-1:] == ["memos"]


def test_centerout_accepts_a_spaced_palindromic_seed_phrase():
    words = centerout_search(
        WordTries(["a", "am", "madam"]),
        ZeroScorer(),
        center="Some men interpret nine memos",
        min_letters=25,
        beam_width=4,
        max_steps=1,
        candidate_limit=16,
    )
    assert "some men interpret nine memos" in " ".join(words).lower()
