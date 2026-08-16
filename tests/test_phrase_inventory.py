"""Tests for building the phrase inventory the search consumes.

`docs/training.md` ends on this: "the trie holds single words, so coherence has
to emerge from adjacent-word scoring. A corpus-derived phrase inventory —
attested n-grams consumed atomically — would make local coherence a property of
each unit rather than something the scorer has to discover."

Two things the inventory must not do, and they are what these tests are for.
It must not smuggle in words the vocabulary filter excluded — a phrase is two
words and both have to pass — and it must not offer a phrase the search cannot
print, so every unit has to survive a round trip through the letters.
"""
import pytest

from llm_palindrome.phrases import build_inventory, parse_bigram_file
from llm_palindrome.search import unit_letters


COUNTS = """new york\t1000
york city\t800
the damn\t900
new jersey\t50
of the\t5000
zzz qqq\t10
"""


@pytest.fixture
def counts_file(tmp_path):
    p = tmp_path / "count_2w.txt"
    p.write_text(COUNTS)
    return str(p)


class TestParseBigramFile:
    def test_reads_pairs_and_counts(self, counts_file):
        pairs = parse_bigram_file(counts_file)
        assert pairs[("new", "york")] == 1000

    def test_skips_malformed_lines(self, tmp_path):
        p = tmp_path / "bad.txt"
        p.write_text("good pair\t5\nnotabigram\t9\nthree word phrase\t7\n")
        pairs = parse_bigram_file(str(p))
        assert list(pairs) == [("good", "pair")]


class TestBuildInventory:
    def test_orders_phrases_by_count(self, counts_file):
        inv = build_inventory(counts_file, vocab={"new", "york", "city", "of", "the"},
                              top_n=2)
        assert inv == ["of the", "new york"]

    def test_drops_a_phrase_whose_second_word_is_not_in_vocab(self, counts_file):
        inv = build_inventory(counts_file, vocab={"new"}, top_n=10)
        assert inv == []

    def test_drops_a_phrase_whose_first_word_is_not_in_vocab(self, counts_file):
        inv = build_inventory(counts_file, vocab={"york", "city"}, top_n=10)
        assert "new york" not in inv

    def test_a_word_blocked_by_the_vocabulary_filter_cannot_return_in_a_phrase(
            self, counts_file):
        """The filter exists because the frequency list carries slurs.

        A phrase inventory drawn from the same corpus is a second door into the
        same vocabulary, and it has to be locked the same way.
        """
        inv = build_inventory(counts_file, vocab={"the"}, top_n=10)
        assert not any("damn" in unit for unit in inv)

    def test_every_unit_is_two_words(self, counts_file):
        inv = build_inventory(counts_file, vocab={"new", "york", "city", "of", "the"},
                              top_n=10)
        assert all(len(unit.split()) == 2 for unit in inv)

    def test_every_unit_survives_the_letters_round_trip(self, counts_file):
        inv = build_inventory(counts_file, vocab={"new", "york", "city", "of", "the"},
                              top_n=10)
        assert all(unit_letters(u).isalpha() and unit_letters(u) for u in inv)

    def test_respects_min_count(self, counts_file):
        inv = build_inventory(counts_file, vocab={"new", "york", "jersey"},
                              top_n=10, min_count=100)
        assert "new jersey" not in inv and "new york" in inv


class TestCorpusNgrams:
    """Bigram units did not produce sentences: two generations, 0/25 judged
    coherent. A two-word unit guarantees one attested join, and a sentence
    needs five or six in a row.

    Longer units have to be mined from running text rather than a pair table,
    and the guarantee they carry is stronger: a 6-gram that occurs in a corpus
    IS a fragment of real English, not an inference from pair counts.
    """

    def test_mines_repeated_ngrams_from_text(self):
        from llm_palindrome.phrases import mine_ngrams
        corpus = ["the cat sat on the mat", "the cat sat on the mat again"]
        found = mine_ngrams(corpus, n=4, min_count=2, vocab=None)
        assert "the cat sat on" in found

    def test_ignores_ngrams_seen_only_once(self):
        from llm_palindrome.phrases import mine_ngrams
        found = mine_ngrams(["a b c d"], n=4, min_count=2, vocab=None)
        assert found == []

    def test_restricts_to_vocab_when_given(self):
        from llm_palindrome.phrases import mine_ngrams
        corpus = ["the cat sat on", "the cat sat on"]
        assert mine_ngrams(corpus, n=4, min_count=2, vocab={"the", "cat", "sat"}) == []

    def test_every_mined_unit_has_exactly_n_words(self):
        from llm_palindrome.phrases import mine_ngrams
        corpus = ["the cat sat on the mat"] * 3
        assert all(len(u.split()) == 5 for u in mine_ngrams(corpus, n=5, min_count=2,
                                                            vocab=None))

    def test_drops_ngrams_carrying_non_alphabetic_tokens(self):
        """Numbers and punctuation cannot go in the palindrome's letters."""
        from llm_palindrome.phrases import mine_ngrams
        corpus = ["born in 1972 in france"] * 3
        assert mine_ngrams(corpus, n=3, min_count=2, vocab=None) == ["in france in"] or \
               all("1972" not in u for u in mine_ngrams(corpus, n=3, min_count=2,
                                                        vocab=None))


class TestMineSentences:
    """Whole sentences, not spans.

    Generation 5 isolated attested n-grams as sentences and the judge rejected
    16 of 16: "Was unable to make.", "Is much too small to be." They are
    grammatical and they trail off, because an n-gram mined from running text
    is a span from the MIDDLE of a sentence.

    A unit that runs from a sentence's start to its end does not have that
    problem, and it needs no repetition to justify it — an n-gram is trusted
    because it recurs, but a sentence occurring once is already a sentence
    somebody wrote.
    """

    def test_takes_a_whole_sentence(self):
        from llm_palindrome.phrases import mine_sentences
        got = mine_sentences(["The film was a success. It made money."],
                             min_words=3, max_words=8, vocab=None)
        assert "the film was a success" in got

    def test_drops_sentences_longer_than_the_cap(self):
        from llm_palindrome.phrases import mine_sentences
        got = mine_sentences(["one two three four five six."], min_words=2,
                             max_words=4, vocab=None)
        assert got == []

    def test_drops_sentences_shorter_than_the_floor(self):
        from llm_palindrome.phrases import mine_sentences
        got = mine_sentences(["Yes."], min_words=3, max_words=8, vocab=None)
        assert got == []

    def test_strips_the_terminating_punctuation(self):
        from llm_palindrome.phrases import mine_sentences
        got = mine_sentences(["The dog ran fast."], min_words=3, max_words=8,
                             vocab=None)
        assert got == ["the dog ran fast"]

    def test_drops_sentences_carrying_non_alphabetic_tokens(self):
        from llm_palindrome.phrases import mine_sentences
        got = mine_sentences(["He was born in 1972."], min_words=3, max_words=8,
                             vocab=None)
        assert got == []

    def test_restricts_to_vocab(self):
        from llm_palindrome.phrases import mine_sentences
        got = mine_sentences(["The dog ran fast."], min_words=3, max_words=8,
                             vocab={"the", "dog", "ran"})
        assert got == []

    def test_does_not_require_repetition(self):
        """A sentence seen once is still a sentence."""
        from llm_palindrome.phrases import mine_sentences
        got = mine_sentences(["A rare thing happened here."], min_words=3,
                             max_words=8, vocab=None)
        assert len(got) == 1
