"""Mirror-pairs built from reversible words, rather than found or authored.

Authoring was tried first and measured: 40 candidates, 40 verified, 37 novel —
but 27 of the 40 were single-word reversals ("step || pets"), the multi-word
ones collapsed to a single family under deduplication, and every four-word half
that verified turned out to be a canon recitation. The model recites, which
iteration 49 already found; the novelty check now shows it at pair level.

What authoring did surface is a template that is not authored at all. For any
word whose reverse is also a word — step/pets, star/rats, bud/dub — the letters
of "step on" reverse to "nopets", so

    step on  ||  no pets

is a mirror-pair with two-word halves, by construction. The connector works
because "on" reverses to "no"; any reversible word can play that part, so the
scheme is L = A B, R = rev(B) rev(A) over a set that can simply be enumerated.

This is the first unit source in the project that is neither searched nor
proposed: it is closed-form. The mirror cost is paid by the rarity of
reversible words, not by a search or a filter.
"""
import pytest

from llm_palindrome.reversibles import pairs_from_reversibles, semordnilaps
from llm_palindrome.validator import is_palindrome

VOCAB = ["step", "pets", "star", "rats", "on", "no", "level", "was", "saw",
         "bud", "dub", "cat", "the", "xyz"]


class TestSemordnilaps:
    def test_finds_a_reversible_pair(self):
        assert ("step", "pets") in semordnilaps(VOCAB)

    def test_excludes_a_word_that_is_its_own_reverse(self):
        """"level" reversed is "level" — a pair of it with itself is a stutter,
        not a mirror."""
        assert ("level", "level") not in semordnilaps(VOCAB)

    def test_excludes_a_word_whose_reverse_is_not_a_word(self):
        assert not any(a == "cat" or b == "cat" for a, b in semordnilaps(VOCAB))

    def test_includes_both_orientations(self):
        """Which half a word opens is the pair builder's choice, not this
        function's."""
        found = semordnilaps(VOCAB)
        assert ("step", "pets") in found and ("pets", "step") in found

    def test_honours_a_minimum_length(self):
        for a, b in semordnilaps(VOCAB, min_letters=4):
            assert len(a) >= 4 and len(b) >= 4

    def test_a_frequency_floor_drops_the_obscure(self):
        """The lexicon carries "tra", "oda", "ria", "lac" and "bom"; every one
        is the reverse of a common word and none is one a reader accepts.

        They sit at zipf 3.0-3.35 while real short words like "dub" sit at
        3.64, so the floor has to be 3.5 — 3.0 keeps all of them.
        """
        common = semordnilaps(["art", "tra", "step", "pets"], min_zipf=3.5)
        assert ("art", "tra") not in common
        assert ("tra", "art") not in common
        assert ("step", "pets") in common


class TestPairsFromReversibles:
    def test_builds_a_two_word_half(self):
        found = {(" ".join(l), " ".join(r))
                 for l, r in pairs_from_reversibles(VOCAB)}
        assert ("step on", "no pets") in found

    def test_every_pair_is_a_palindrome(self):
        for left, right in pairs_from_reversibles(VOCAB):
            assert is_palindrome(" ".join(left) + " " + " ".join(right))

    def test_the_connector_must_itself_be_reversible(self):
        """"on" works because it reverses to "no". A connector that does not
        reverse to a word cannot appear, or the mirror breaks."""
        for left, right in pairs_from_reversibles(VOCAB):
            assert left[-1][::-1] == right[0]

    def test_no_half_repeats_a_word(self):
        for left, right in pairs_from_reversibles(VOCAB):
            assert len(set(left)) == len(left)

    def test_it_yields_more_than_one_connector(self):
        found = {" ".join(l) for l, _ in pairs_from_reversibles(VOCAB)}
        connectors = {phrase.split()[-1] for phrase in found}
        assert len(connectors) > 1

    def test_a_limit_bounds_the_output(self):
        assert len(list(pairs_from_reversibles(VOCAB, limit=3))) <= 3


class TestAgainstTheShippingVocabulary:
    @pytest.mark.slow
    def test_it_beats_mining_on_four_word_halves(self):
        """Mining the corpus produced 0 both-attested four-word halves.

        This is the claim worth checking: a closed-form scheme reaches lengths
        the corpus could not. A half here is A + connector, so length comes
        from stacking connectors.
        """
        from llm_palindrome.generate import build_vocab
        from llm_palindrome.lexicon import is_real_word, load_lexicon

        lexicon = load_lexicon("data/lexicon.txt")
        vocab = [w for w in build_vocab(30000) if is_real_word(w, lexicon)]
        pairs = list(pairs_from_reversibles(vocab, min_zipf=3.0, limit=5000))
        assert pairs
        for left, right in pairs[:200]:
            assert is_palindrome(" ".join(left) + " " + " ".join(right))
