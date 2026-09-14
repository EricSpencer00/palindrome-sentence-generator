from llm_palindrome.search import State, WordTries, beam_search


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
