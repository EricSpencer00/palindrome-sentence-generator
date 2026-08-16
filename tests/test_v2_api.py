"""Tests for the v2 service layer.

v2 differs from v1 in what it puts in the text: whole sentences mined from
Wikipedia, placed atomically. That has two consequences the service has to
handle and v1 never did.

The output QUOTES. A sentence like "Hobbs did not order the saloon closed
down." is somebody else's writing under CC BY-SA, so the response has to say
which spans are quoted and where they came from — a public endpoint that
serves them silently is redistributing Wikipedia without attribution.

And the sentence is the unit the reader cares about, so the response carries
the segmentation rather than leaving the page to guess where one ends.
"""
import os

os.environ.setdefault("PALINDROME_NO_WARM", "1")

from server.v2 import quoted_units, sentence_payload


class TestQuotedUnits:
    def test_finds_a_unit_that_came_from_the_inventory(self):
        assert quoted_units(["oo", "the dog ran fast", "ee"],
                            {"the dog ran fast"}) == ["the dog ran fast"]

    def test_ignores_units_the_search_assembled_itself(self):
        assert quoted_units(["oo", "the dog", "ee"], {"the dog ran fast"}) == []

    def test_reports_each_distinct_quote_once(self):
        got = quoted_units(["a b c", "x", "a b c"], {"a b c"})
        assert got == ["a b c"]

    def test_empty_when_nothing_was_quoted(self):
        assert quoted_units(["oo", "ee"], {"the dog ran fast"}) == []


class TestSentencePayload:
    def test_marks_which_sentences_are_quoted(self):
        payload = sentence_payload(["oo", "the dog ran fast"], {"the dog ran fast"})
        assert [s["quoted"] for s in payload] == [False, True]

    def test_renders_each_sentence_capitalised_and_stopped(self):
        payload = sentence_payload(["the dog ran fast"], {"the dog ran fast"})
        assert payload[0]["text"] == "The dog ran fast."

    def test_groups_filler_between_quotes_into_one_sentence(self):
        payload = sentence_payload(["oo", "ee", "the dog ran fast", "aa"],
                                   {"the dog ran fast"})
        assert [s["text"] for s in payload] == ["Oo ee.", "The dog ran fast.", "Aa."]

    def test_the_letters_are_never_altered(self):
        from llm_palindrome.validator import normalize
        units = ["oo", "the dog ran fast", "aa", "bb"]
        payload = sentence_payload(units, {"the dog ran fast"})
        rendered = " ".join(s["text"] for s in payload)
        assert normalize(rendered) == normalize(" ".join(units))

    def test_no_quotes_still_produces_sentences(self):
        payload = sentence_payload(["oo", "ee"], set())
        assert payload and all(s["quoted"] is False for s in payload)


class TestSeedVaries:
    """A pinned seed serves every visitor the same palindrome.

    v1 pins seed=0 and gets away with it because its output is anonymous
    rubble. v2's is memorable — the same five Wikipedia sentences in the same
    order — so a fixed seed turns the endpoint into a static page that takes
    twelve seconds to load.
    """

    def test_the_same_prompt_gives_different_seeds_on_different_calls(self):
        from server.v2 import request_seed
        seeds = {request_seed("hello", nonce=i) for i in range(5)}
        assert len(seeds) == 5

    def test_the_seed_is_a_non_negative_int(self):
        from server.v2 import request_seed
        s = request_seed("hello", nonce=0)
        assert isinstance(s, int) and s >= 0


class TestRetrySeeds:
    """A random seed is what makes visitors get different palindromes, and it
    is also what makes some of them get nothing: at the 400-letter floor, 17 of
    20 random seeds close. v1 never had to care because it pinned seed 0.

    One search takes about two seconds against a twelve-second budget, so the
    fix is to try again rather than to shorten the text or serve everyone the
    same thing.
    """

    def test_yields_distinct_seeds(self):
        from server.v2 import seed_sequence
        seeds = seed_sequence("hi", nonce=1234, attempts=4)
        assert len(set(seeds)) == 4

    def test_yields_exactly_the_requested_number(self):
        from server.v2 import seed_sequence
        assert len(seed_sequence("hi", nonce=1, attempts=3)) == 3

    def test_a_different_nonce_gives_a_different_run(self):
        from server.v2 import seed_sequence
        assert seed_sequence("hi", nonce=1, attempts=3) != seed_sequence("hi", nonce=2,
                                                                         attempts=3)
