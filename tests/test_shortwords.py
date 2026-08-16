"""Tests for which one- and two-letter words the search may use.

The output was 52.6% one- and two-letter words against real English's 18.5%,
and the excess was not English at all: bn, cu, eb, ek, fo, ac, iw, ht. Those
entered with the 100k vocabulary — wordfreq ranks by corpus frequency and the
web says "bn" often enough to make the cut.

Short words are the search's favourite filler because they fit any overhang, so
this is the one place in the vocabulary where an explicit list beats a
frequency threshold. Brown at 20+ occurrences still admits "aj", "du" and every
bare initial; a threshold cannot tell an English word from a common typo.
"""
from llm_palindrome.shortwords import REAL_SHORT_WORDS, is_real_short


class TestRealShortWords:
    def test_accepts_ordinary_english_short_words(self):
        for w in ("a", "i", "of", "to", "in", "is", "it", "be", "we", "no"):
            assert is_real_short(w), w

    def test_rejects_the_fragments_the_generator_was_emitting(self):
        for w in ("bn", "cu", "eb", "ek", "fo", "ac", "iw", "ht", "tu", "ob"):
            assert not is_real_short(w), w

    def test_rejects_bare_initials(self):
        for w in ("b", "c", "d", "q", "v", "z"):
            assert not is_real_short(w), w

    def test_longer_words_are_not_its_business(self):
        assert is_real_short("elephant")

    def test_the_list_is_small_enough_to_audit(self):
        short = {w for w in REAL_SHORT_WORDS if len(w) <= 2}
        assert len(short) < 60

    def test_every_listed_word_is_short(self):
        assert all(len(w) <= 2 for w in REAL_SHORT_WORDS)


class TestVocabularyIsFiltered:
    def test_build_vocab_excludes_short_non_words(self):
        from llm_palindrome.generate import build_vocab
        vocab = set(build_vocab(100000))
        assert not ({"bn", "cu", "eb", "ek", "fo", "iw", "ht"} & vocab)

    def test_build_vocab_keeps_real_short_words(self):
        from llm_palindrome.generate import build_vocab
        vocab = set(build_vocab(100000))
        assert {"of", "to", "in", "is", "it", "a", "i"} <= vocab

    def test_build_vocab_still_returns_long_words(self):
        from llm_palindrome.generate import build_vocab
        vocab = build_vocab(30000)
        assert sum(1 for w in vocab if len(w) > 4) > 10000
