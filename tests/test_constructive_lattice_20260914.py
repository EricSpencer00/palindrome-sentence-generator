from llm_palindrome.centerout import COState, centerout_search
from llm_palindrome.search import WordTries


class ZeroScorer:
    def word_delta(self, left, right, placement, word, growth):
        return 0.0


def test_centerout_allow_word_gate_runs_before_child_scoring():
    seen = []

    def allow(placement, word, state):
        seen.append(word)
        return word == "memos"

    result = centerout_search(
        WordTries(["memos", "some", "men", "a"]), ZeroScorer(),
        min_letters=1, beam_width=8, max_steps=1, candidate_limit=32,
        allow_word=allow,
    )
    assert seen
    assert result == []


def test_centerout_can_continue_from_a_closed_authored_endpoint():
    state = COState(
        0.0, ("an", "aide", "rips", "nine", "memos"),
        ("some", "men", "inspire", "diana"), "", "R", 0.0,
    )
    result = centerout_search(
        WordTries(["memos", "some", "men", "inspire", "diana", "an", "aide", "rips", "nine"]),
        ZeroScorer(), min_letters=30, beam_width=4, max_steps=1,
        candidate_limit=32, initial_state=state,
    )
    assert result == list(state.left + state.right)


def test_lattice_audit_records_exactness_without_claiming_readability():
    from experiments.brown_shape_lattice_exact_search_20260914 import audit

    row = audit((
        ("an", "aide", "rips", "nine", "memos"),
        ("some", "men", "inspire", "diana"),
    ))
    assert row["checks"]["independent_exact_audit"]
    assert row["checks"]["shape_pair_reverse"]
    assert row["reader_status"] == "not_run"
