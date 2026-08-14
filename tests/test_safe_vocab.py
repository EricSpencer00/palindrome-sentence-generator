"""The vocabulary feeds a public endpoint, so it has to be clean.

wordfreq ranks words by how often they appear on the web, which puts porn and
slur vocabulary well inside the top 30k. The search has no notion of taste and
the bigram model will happily chain those words into fluent-sounding phrases,
so the filtering has to happen at the vocabulary, before search ever sees them.
"""
from llm_palindrome.generate import build_vocab
from llm_palindrome.safe_vocab import is_allowed, safe_vocab


class TestIsAllowed:
    def test_rejects_explicit_terms(self):
        for w in ("rape", "xxx", "porn", "bitch"):
            assert not is_allowed(w), f"{w!r} should be filtered"

    def test_rejects_slurs(self):
        assert not is_allowed("nigger")
        assert not is_allowed("faggot")

    def test_keeps_ordinary_words(self):
        for w in ("never", "odd", "even", "table", "matter", "class"):
            assert is_allowed(w), f"{w!r} should survive"

    def test_keeps_innocent_substrings(self):
        """Substring matching would eat these; the filter is word-level."""
        for w in ("assist", "class", "grass", "analysis", "shitake", "cocktail"):
            assert is_allowed(w), f"{w!r} wrongly filtered"


def _raw_wordfreq_vocab():
    """The unfiltered source build_vocab draws from."""
    from wordfreq import top_n_list
    return [w for w in top_n_list("en", 30000) if w.isalpha() and w.isascii()]


class TestSafeVocab:
    def test_filters_the_raw_wordfreq_list(self):
        raw = _raw_wordfreq_vocab()
        clean = safe_vocab(raw)
        assert len(clean) < len(raw), "nothing was filtered"
        for w in ("rape", "xxx", "porn"):
            assert w not in clean
        assert "never" in clean and "even" in clean

    def test_removes_at_most_a_small_share(self):
        raw = _raw_wordfreq_vocab()
        removed = 1 - len(safe_vocab(raw)) / len(raw)
        assert removed < 0.05, f"filter is too aggressive: removed {removed:.1%}"

    def test_build_vocab_is_already_clean(self):
        """The filter has to be applied at the source, not left to callers."""
        v = set(build_vocab())
        for w in ("rape", "xxx", "porn", "bitch", "nigger", "sex"):
            assert w not in v, f"{w!r} reached the public vocabulary"
