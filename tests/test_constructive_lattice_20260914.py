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


def test_centerout_counts_letters_not_spaces_in_a_fixed_center():
    from llm_palindrome.centerout import centerout_search
    from llm_palindrome.validator import normalize

    center = "an aide rips nine memos some men inspire diana"
    allowed = {"to", "no", "set", "is", "site", "so", "not"}

    def allow(placement, word, state):
        return word in allowed

    # This branch closes at 56 letters. A spaced-center accounting bug used to
    # count the 9 spaces as letters and admit it under a 60-letter floor.
    result = centerout_search(
        WordTries(sorted(allowed)), ZeroScorer(), center=center,
        min_letters=60, beam_width=24, max_steps=120, candidate_limit=96,
        maximize="letters", allow_word=allow,
    )
    assert result == [] or len(normalize(" ".join(result))) >= 60


def test_fixed_center_extension_audit_independently_checks_the_seed():
    from tools.polaris.center_extension_debug import CENTER, exact_audit

    text = "an aide rips nine memos some men inspire diana"
    audit = exact_audit(text, CENTER)
    assert audit["independent_exact"]
    assert audit["validator_exact"]
    assert audit["letters"] == 38
