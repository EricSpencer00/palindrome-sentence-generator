"""Mine mirror-pairs from attested English phrases.

The canon gives 23 readable pairs, which is not enough material for a
paragraph. The hunts give 20,000 unreadable ones, because an exhaustive walk
over a vocabulary proposes "nora aaron" as readily as "no lemon" and the LM
ranking cannot tell them apart (README: every proxy tried failed against judge
verdicts).

The way out is to stop proposing. A left half taken from an attested English
bigram is readable because English attested it; the only open question is
whether its mirror reads too, and `respace` answers that by segmentation. Over
272k attested bigrams this yields tens of thousands of pairs in seconds —
"went on || not new", "test on || not set", "emits a || as time".

SAFETY: `respace.canon_vocab` deliberately omits `safe_vocab`, on the grounds
that recovering the spelling of an already-stored palindrome is not generation.
Mining IS generation — it invents phrases this project has never held — so it
must use the filtered vocabulary. A trial run with the unfiltered one put
"not raped" in the output, which is precisely what the filter exists to stop.
"""
import pytest

from llm_palindrome.mining import mine_pairs
from llm_palindrome.validator import is_palindrome

VOCAB = ["no", "pets", "step", "on", "not", "new", "went", "test", "set",
         "evil", "live", "star", "rats", "level", "a", "of", "for", "even",
         "never"]

PHRASES = ["step on", "went on", "test on", "no evil", "for even", "level a"]


class TestMinePairs:
    def test_every_pair_is_a_palindrome(self):
        for left, right in mine_pairs(PHRASES, VOCAB):
            assert is_palindrome(" ".join(left) + " " + " ".join(right))

    def test_it_finds_the_obvious_ones(self):
        found = {(" ".join(l), " ".join(r)) for l, r in mine_pairs(PHRASES, VOCAB)}
        assert ("step on", "no pets") in found
        assert ("went on", "not new") in found

    def test_output_words_all_come_from_the_vocabulary(self):
        for left, right in mine_pairs(PHRASES, VOCAB):
            assert all(w in VOCAB for w in left)
            assert all(w in VOCAB for w in right)

    def test_no_duplicate_pairs(self):
        out = [(" ".join(l), " ".join(r))
               for l, r in mine_pairs(PHRASES * 3, VOCAB)]
        assert len(out) == len(set(out))

    def test_a_phrase_using_words_outside_the_vocabulary_is_skipped(self):
        assert not list(mine_pairs(["quokka sprints"], VOCAB))

    def test_respects_the_letter_bounds(self):
        for left, _ in mine_pairs(PHRASES, VOCAB, min_letters=8,
                                  max_letters=10):
            assert 8 <= len("".join(left)) <= 10

    def test_can_require_multi_word_halves(self):
        """One-word halves make the paragraph a list of nouns."""
        for left, right in mine_pairs(PHRASES, VOCAB, min_words=2):
            assert len(left) >= 2 and len(right) >= 2

    def test_can_forbid_one_letter_words(self):
        for left, right in mine_pairs(PHRASES, VOCAB, min_word_letters=2):
            assert min(len(w) for w in list(left) + list(right)) >= 2

    def test_a_pair_that_mirrors_onto_itself_is_dropped(self):
        """"level a || a level" adds nothing a reader can hear as a turn."""
        out = {(" ".join(l), " ".join(r)) for l, r in mine_pairs(PHRASES, VOCAB)}
        assert ("level a", "a level") not in out


class TestGenerationSafety:
    """The filter that a trial run defeated. See this module's docstring."""

    def test_a_blocked_word_never_reaches_a_left_half(self):
        from llm_palindrome.safe_vocab import is_allowed

        assert not is_allowed("raped"), "fixture assumes this stays blocked"
        vocab = [w for w in VOCAB + ["raped", "depar"] if is_allowed(w)]
        for left, right in mine_pairs(["not raped"], vocab):
            assert "raped" not in list(left) + list(right)

    @pytest.mark.slow
    def test_mining_the_real_corpus_emits_nothing_blocked(self):
        """The end-to-end guarantee, on the vocabulary that actually ships."""
        from llm_palindrome.generate import build_vocab
        from llm_palindrome.mining import attested_phrases
        from llm_palindrome.safe_vocab import is_allowed

        vocab = build_vocab(30000)
        phrases = list(attested_phrases("data/count_2w.txt", vocab))[:40000]
        for left, right in mine_pairs(phrases, vocab, min_words=2):
            for w in list(left) + list(right):
                assert is_allowed(w), w


class TestMiningTheOtherDirection:
    """Mining a phrase as the right half instead of the left.

    This was built to recover pairs lost to `respace` returning one reading,
    and it does NOT do that: measured over the corpus, every one of the 7,704
    entries came back with its own flip and no new material. Swapping which
    half is "given" cannot change which readings the model produces — if L is
    attested and respace(mirror(L)) is R, feeding L in as a right half just
    emits (R, L). k-best is what recovers those pairs; see test_respace.py.

    The option is kept because a pair genuinely is usable either way round and
    the seams it makes differ, so sequencing can want both orientations. It is
    not a source of new units, and the inventory must not double-count it.
    """

    def test_a_phrase_can_be_mined_as_the_right_half(self):
        found = {(" ".join(l), " ".join(r))
                 for l, r in mine_pairs(["no pets"], VOCAB, side="right")}
        assert ("step on", "no pets") in found

    def test_the_attested_half_lands_on_the_side_asked_for(self):
        for left, right in mine_pairs(["went on"], VOCAB, side="right"):
            assert " ".join(right) == "went on"

    def test_both_directions_still_produce_palindromes(self):
        for side in ("left", "right"):
            for left, right in mine_pairs(PHRASES, VOCAB, side=side):
                assert is_palindrome(" ".join(left) + " " + " ".join(right))

    def test_an_unknown_side_is_rejected(self):
        with pytest.raises(ValueError):
            list(mine_pairs(PHRASES, VOCAB, side="middle"))

    def test_the_other_direction_only_ever_returns_flips(self):
        """The measured no-op, pinned so nobody re-derives the hope.

        Anything mined with the phrase on the right is the flip of something
        mined with it on the left. The union is exactly a doubling.
        """
        left = {(" ".join(l), " ".join(r))
                for l, r in mine_pairs(PHRASES, VOCAB, side="left")}
        right = {(" ".join(l), " ".join(r))
                 for l, r in mine_pairs(PHRASES, VOCAB, side="right")}
        assert right, "fixture produced nothing to compare"
        assert all((r, l) in left for l, r in right)


class TestBothHalvesAttested:
    """The strongest readability evidence available, and it is rare.

    Mining guarantees the LEFT half reads, because English attested it. The
    right half is only filtered — real words, in some order — which is why
    "eta it in it on" survives. When the right half is ALSO an attested phrase,
    both directions read and the mirror cost is fully paid.

    Measured over 3,922 mined pairs: 157 have an attested right half, 129 of
    them multi-word. That 3% is the cost, made visible.
    """

    def test_a_fully_attested_half_passes(self):
        from llm_palindrome.mining import reads_as_attested

        assert reads_as_attested(["not", "up"], {("not", "up")})

    def test_an_unattested_join_fails(self):
        from llm_palindrome.mining import reads_as_attested

        assert not reads_as_attested(["eta", "it"], {("not", "up")})

    def test_a_single_word_half_has_no_join_to_check(self):
        from llm_palindrome.mining import reads_as_attested

        assert reads_as_attested(["fillet"], set())

    def test_every_join_must_be_attested_not_just_one(self):
        from llm_palindrome.mining import reads_as_attested

        attested = {("no", "it"), ("it", "can")}
        assert reads_as_attested(["no", "it", "can"], attested)
        assert not reads_as_attested(["no", "it", "zzz"], attested)

    def test_mining_can_prefer_an_attested_reading(self):
        """Given the choice of readings, take the one English attests."""
        vocab = VOCAB + ["award", "ward", "draw"]
        attested = {("a", "ward")}
        found = {" ".join(r) for _, r in
                 mine_pairs(["draw a"], vocab, min_letters=4,
                            prefer_attested=attested)}
        assert "a ward" in found

    def test_a_one_word_reading_does_not_win_on_vacuous_attestation(self):
        """"award" has no joins, so `reads_as_attested` is trivially true.

        Taking that as evidence made the single word beat "a ward" — the
        reading English actually attests, and the only one that survives a
        min_words of 2.
        """
        vocab = VOCAB + ["award", "ward", "draw"]
        found = {" ".join(r) for _, r in
                 mine_pairs(["draw a"], vocab, min_letters=4,
                            prefer_attested={("a", "ward")})}
        assert "award" not in found

    def test_without_the_preference_it_takes_the_best_reading(self):
        vocab = VOCAB + ["award", "ward", "draw"]
        found = {" ".join(r) for _, r in
                 mine_pairs(["draw a"], vocab, min_letters=4)}
        assert found == {"award"}

    def test_attested_bigrams_reads_the_count_file(self, tmp_path):
        from llm_palindrome.mining import attested_bigrams

        p = tmp_path / "counts.txt"
        p.write_text("Not Up\t50\nsolo\t9\n")
        got = attested_bigrams(str(p))
        assert ("not", "up") in got, "case must be normalised"
        assert len(got) == 1


class TestAttestedNgrams:
    """Three-word phrases, because two-word ones cap the paragraph at "Not as".

    A half can only be as long as the phrase it came from, so an inventory
    mined from bigrams alone is an inventory of two-word fragments. The
    WikiText n-gram file already in data/ carries 93k attested trigrams.

    The yield falls off exactly as the mirror cost predicts — 131 both-attested
    pairs from bigrams, 27 from trigrams, 0 from 4-grams — so trigrams are
    worth mining and 4-grams are not.
    """

    def test_reads_the_requested_length(self, tmp_path):
        from llm_palindrome.mining import attested_ngrams

        p = tmp_path / "ngrams.json"
        p.write_text('{"3": ["it is a", "was in a"], "4": ["at the end of"]}')
        got = list(attested_ngrams(str(p), ["it", "is", "a", "was", "in"], 3))
        assert got == ["it is a", "was in a"]

    def test_skips_phrases_with_words_outside_the_vocabulary(self, tmp_path):
        from llm_palindrome.mining import attested_ngrams

        p = tmp_path / "ngrams.json"
        p.write_text('{"3": ["it is a", "quokkas sprint fast"]}')
        got = list(attested_ngrams(str(p), ["it", "is", "a"], 3))
        assert got == ["it is a"]

    def test_a_missing_length_is_empty_not_an_error(self, tmp_path):
        from llm_palindrome.mining import attested_ngrams

        p = tmp_path / "ngrams.json"
        p.write_text('{"3": ["it is a"]}')
        assert list(attested_ngrams(str(p), ["it", "is", "a"], 9)) == []

    def test_it_normalises_case(self, tmp_path):
        from llm_palindrome.mining import attested_ngrams

        p = tmp_path / "ngrams.json"
        p.write_text('{"3": ["It Is A"]}')
        assert list(attested_ngrams(str(p), ["it", "is", "a"], 3)) == ["it is a"]


class TestAttestedPhrases:
    def test_reads_the_count_file(self, tmp_path):
        from llm_palindrome.mining import attested_phrases

        p = tmp_path / "counts.txt"
        p.write_text("step on\t500\nquokka sprints\t9\nNo Evil\t7\n")
        got = list(attested_phrases(str(p), VOCAB))
        assert "step on" in got
        assert "no evil" in got, "case must be normalised"
        assert "quokka sprints" not in got

    def test_orders_by_attestation(self, tmp_path):
        from llm_palindrome.mining import attested_phrases

        p = tmp_path / "counts.txt"
        p.write_text("no evil\t5\nstep on\t900\n")
        assert list(attested_phrases(str(p), VOCAB))[0] == "step on"
