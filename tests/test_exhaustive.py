"""Tests for exhaustive enumeration of short palindromes.

Every search in this project is a beam: it keeps the best `k` states and throws
the rest away, which is right when the target is long and wrong when the target
is short. The readable palindromes in the record are 24 to 30 letters, and at
that size the whole space can be walked — no scorer, no beam, no discarding.
What a scorer is for afterwards is choosing among everything that exists,
rather than steering toward a corner of it.

The enumeration is the part that has to be correct: a beam that misses a good
branch produces worse text, but an enumeration that misses one is not
exhaustive and the word means nothing.
"""
import pytest

from llm_palindrome.exhaustive import enumerate_palindromes
from llm_palindrome.search import WordTries
from llm_palindrome.validator import is_palindrome, normalize


VOCAB = ["a", "man", "plan", "canal", "panama", "no", "on", "step", "pets",
         "rats", "star", "live", "evil", "was", "saw", "it", "i", "car", "rac"]


def run(max_letters=14, **kw):
    return list(enumerate_palindromes(WordTries(VOCAB), max_letters=max_letters, **kw))


class TestEnumerationIsSound:
    def test_every_result_is_a_palindrome(self):
        out = run()
        assert out and all(is_palindrome(" ".join(u)) for u in out)

    def test_no_result_exceeds_the_letter_budget(self):
        for u in run(max_letters=10):
            assert len(normalize(" ".join(u))) <= 10

    def test_results_are_word_sequences_from_the_vocabulary(self):
        allowed = set(VOCAB)
        for u in run():
            assert all(w in allowed for w in u)

    def test_a_minimum_length_is_respected(self):
        for u in run(max_letters=14, min_letters=8):
            assert len(normalize(" ".join(u))) >= 8


class TestEnumerationIsComplete:
    def test_it_finds_a_known_palindrome(self):
        """'step on no pets' is reachable and must actually be reached."""
        out = {" ".join(u) for u in run(max_letters=12)}
        assert "step on no pets" in out

    def test_it_finds_another_known_one(self):
        """20 letters, so the budget has to be at least that."""
        out = {" ".join(u) for u in run(max_letters=20)}
        assert "rats live on no evil star" in out

    def test_raising_the_budget_never_loses_a_result(self):
        small = {" ".join(u) for u in run(max_letters=10)}
        large = {" ".join(u) for u in run(max_letters=14)}
        assert small <= large


class TestSharding:
    def test_shards_partition_the_results(self):
        whole = {" ".join(u) for u in run(max_letters=12)}
        parts = set()
        for i in range(3):
            parts |= {" ".join(u) for u in run(max_letters=12, shard=i, shards=3)}
        assert parts == whole

    def test_shards_do_not_overlap(self):
        seen = [{" ".join(u) for u in run(max_letters=12, shard=i, shards=3)}
                for i in range(3)]
        assert not (seen[0] & seen[1]) and not (seen[1] & seen[2])


class TestBudget:
    def test_a_node_budget_stops_it_early(self):
        few = run(max_letters=16, node_budget=50)
        many = run(max_letters=16, node_budget=10**7)
        assert len(few) < len(many)


class TestAcceptanceFilter:
    """An exhaustive walk finds every degenerate closure the vocabulary allows.

    A first run returned "ann aaa aaron nora aaa anna" as its best result: "aaa"
    is in the frequency list because the web says it, and it fits any overhang,
    so the walk builds with it. Filtering before the GPU is what keeps the job
    from spending an allocation scoring filler.
    """

    def _zipf(self, w):
        return 5.0

    def test_rejects_a_word_that_is_one_letter_repeated(self):
        from llm_palindrome.exhaustive import acceptable_words
        assert not acceptable_words(["aaa", "star", "rats", "well"])

    def test_keeps_ordinary_words(self):
        from llm_palindrome.exhaustive import acceptable_words
        assert acceptable_words(["rats", "live", "on", "no", "evil", "star"])

    def test_rejects_text_that_is_mostly_tiny_words(self):
        from llm_palindrome.exhaustive import acceptable_words
        assert not acceptable_words(["a", "an", "na", "a"], min_mean_len=3.0)

    def test_a_single_letter_word_is_still_allowed_in_context(self):
        from llm_palindrome.exhaustive import acceptable_words
        assert acceptable_words(["step", "on", "no", "pets"], min_mean_len=3.0)


class TestHuntVocabulary:
    """Prune the vocabulary, not the results.

    The first walk spent its whole node budget inside the "aaa" subtree: the
    trie sorts its units, the DFS is LIFO, and "aaa" fits every overhang. Every
    closure it found was filtered out afterwards, which means the budget bought
    nothing. A word that can never appear in an acceptable result should not be
    in the trie at all.
    """

    def test_drops_repeated_letter_words(self):
        from llm_palindrome.exhaustive import hunt_vocabulary
        assert "aaa" not in hunt_vocabulary(["aaa", "star", "rats"], lambda w: 9.0)

    def test_drops_words_below_the_frequency_floor(self):
        from llm_palindrome.exhaustive import hunt_vocabulary
        z = {"star": 5.0, "zyzzyva": 1.0}
        out = hunt_vocabulary(["star", "zyzzyva"], z.get, min_zipf=3.0)
        assert out == ["star"]

    def test_keeps_ordinary_words(self):
        from llm_palindrome.exhaustive import hunt_vocabulary
        out = hunt_vocabulary(["star", "rats", "on", "no"], lambda w: 5.0)
        assert set(out) == {"star", "rats", "on", "no"}


class TestTimeBudget:
    """A queue job is bounded by walltime, so the walk must be too.

    The node budget stopped nothing in 55 minutes: at the measured walk rate,
    40M nodes per shard is ~2.8 hours, so both Polaris runs were killed with
    empty output. Each worker now walks until a deadline and returns what it
    has — exhaustive within a time budget, stated as such.
    """

    def test_a_deadline_stops_the_walk(self):
        import time
        from llm_palindrome.exhaustive import enumerate_palindromes
        from llm_palindrome.search import WordTries
        tries = WordTries(VOCAB)
        t0 = time.time()
        list(enumerate_palindromes(tries, max_letters=30, max_units=30,
                                   node_budget=10**9,
                                   deadline=time.time() + 0.3))
        assert time.time() - t0 < 2.0

    def test_no_deadline_means_no_time_limit(self):
        from llm_palindrome.exhaustive import enumerate_palindromes
        from llm_palindrome.search import WordTries
        out = list(enumerate_palindromes(WordTries(VOCAB), max_letters=12))
        assert out  # completes on its own


class TestFrontierOrder:
    """A LIFO walk under a time budget is not a sample of the space.

    The walk returned 2.55M palindromes and none of the 27 canonical ones —
    including "rats live on no evil star", which the enumerator produces
    instantly on a small vocabulary and whose every word is in the hunt
    vocabulary. With ~14k units the depth-first frontier drills into its first
    openings and the budget expires before it ever backtracks, so what came
    back was a deep prefix of one corner, not coverage.
    """

    def test_shuffled_frontier_reaches_a_target_a_lifo_walk_buries(self):
        from llm_palindrome.exhaustive import enumerate_palindromes
        from llm_palindrome.search import WordTries
        from llm_palindrome.validator import normalize
        # 'aaa'-style filler first in sort order buries the real target in DFS.
        vocab = ["ada", "aha", "ana", "step", "on", "no", "pets"]
        tries = WordTries(vocab)
        target = "steponnopets"

        def found(**kw):
            return any(normalize(" ".join(u)) == target
                       for u in enumerate_palindromes(tries, max_letters=12,
                                                      min_letters=12,
                                                      node_budget=4000, **kw))
        assert found(shuffle_seed=0)

    def test_shuffle_seed_changes_the_order_explored(self):
        from llm_palindrome.exhaustive import enumerate_palindromes
        from llm_palindrome.search import WordTries
        tries = WordTries(["ada", "aha", "step", "on", "no", "pets", "ana"])
        a = [" ".join(u) for u in enumerate_palindromes(
            tries, max_letters=12, node_budget=300, shuffle_seed=1)]
        b = [" ".join(u) for u in enumerate_palindromes(
            tries, max_letters=12, node_budget=300, shuffle_seed=2)]
        assert a != b

    def test_default_is_unchanged_deterministic(self):
        from llm_palindrome.exhaustive import enumerate_palindromes
        from llm_palindrome.search import WordTries
        tries = WordTries(["step", "on", "no", "pets"])
        a = [" ".join(u) for u in enumerate_palindromes(tries, max_letters=12)]
        b = [" ".join(u) for u in enumerate_palindromes(tries, max_letters=12)]
        assert a == b


class TestJoinConstraint:
    """`allow_join` prunes the walk instead of filtering its output.

    The distinction is the whole reason it lives in the enumerator: a
    requirement that every adjacency be one English attests rejects almost
    every closure the walk produces, and paying for those closures first is
    what makes an attested-join hunt unaffordable as a post-filter.
    """

    def test_refusing_every_join_leaves_one_word_a_side(self):
        """A half cannot grow past its first word, so nothing exceeds two."""
        out = run(max_letters=14, allow_join=lambda a, b: False)
        assert out and all(len(words) <= 2 for words in out)

    def test_every_join_inside_a_half_was_allowed(self):
        from llm_palindrome.pairs import split_at_mirror

        allowed = {("step", "on"), ("no", "pets"), ("was", "it"),
                   ("it", "i"), ("i", "saw"), ("a", "man"), ("no", "on")}

        for words in run(max_letters=16,
                         allow_join=lambda a, b: (a, b) in allowed):
            split = split_at_mirror(words)
            if split is None:      # the mirror runs through a word: a centre
                continue
            for half in split:
                for join in zip(half, half[1:]):
                    assert join in allowed, (words, join)

    def test_it_still_finds_a_palindrome_whose_joins_are_all_allowed(self):
        allowed = {("step", "on"), ("no", "pets")}
        out = run(max_letters=14, allow_join=lambda a, b: (a, b) in allowed)
        assert ["step", "on", "no", "pets"] in out

    def test_no_constraint_is_the_previous_behaviour(self):
        assert run(max_letters=12) == run(max_letters=12, allow_join=None)


class TestJoinSlack:
    """A budget of refused joins, because English makes new ones all day."""

    def test_slack_admits_a_branch_the_constraint_refused(self):
        allowed = {("step", "on")}

        def ok(before, after):
            return (before, after) in allowed

        strict = run(max_letters=14, allow_join=ok)
        loose = run(max_letters=14, allow_join=ok, join_slack=1)
        assert len(loose) > len(strict)

    def test_the_budget_is_spent_not_refreshed(self):
        """One slack cannot buy two refused joins in the same branch."""
        from llm_palindrome.pairs import split_at_mirror

        allowed = {("step", "on")}
        for words in run(max_letters=20, allow_join=lambda a, b: (a, b) in allowed,
                         join_slack=1):
            split = split_at_mirror(words)
            if split is None:
                continue
            refused = sum((a, b) not in allowed
                          for half in split for a, b in zip(half, half[1:]))
            assert refused <= 1, words

    def test_no_slack_is_the_previous_behaviour(self):
        ok = lambda a, b: (a, b) in {("step", "on")}
        assert run(max_letters=14, allow_join=ok) == run(
            max_letters=14, allow_join=ok, join_slack=0)
