"""Recover the readable spelling of a letters-only palindrome.

`data/known_palindromes.json` stores the canon normalised — "amanaplanacanal
panama" — which is right for the novelty check, where spacing is noise. It is
fatal everywhere else. `harvest` splits a palindrome into a mirror-pair at the
word boundary nearest its midpoint, so a text with no word boundaries can only
ever become a *centre*: the canon yields 0 pairs and 120 centres, and a
paragraph takes exactly one centre.

That is why the letter-level assembler has never been given readable material.
The 20,000 harvested pairs come from the hunts, whose filters admit "nora
aaron" and "bros bacteria"; the 120 palindromes that actually read as English
were unusable for a reason that is purely a storage format.

Recovery is word segmentation under a unigram model — the same problem the
right half of every mirror-pair poses. These tests measure it against the ten
classics whose spelling is recorded in runs/classics_control.json.
"""
import json
from pathlib import Path

import pytest

from llm_palindrome.respace import respace, unigram_score

CLASSICS = {
    "asantalivedasadevilatnasa": "a santa lived as a devil at nasa",
    "amanaplanacanalpanama": "a man a plan a canal panama",
    "neveroddoreven": "never odd or even",
    "nolemonnomelon": "no lemon no melon",
    "madaminedenimadam": "madam in eden im adam",
    "wasitacaroracatisaw": "was it a car or a cat i saw",
    "dogeeseseegod": "do geese see god",
    "steponnopets": "step on no pets",
    "ratsliveonnoevilstar": "rats live on no evil star",
    "siridemandiamamaidnamediris": "sir i demand i am a maid named iris",
}


@pytest.fixture(scope="module")
def vocab():
    from llm_palindrome.respace import canon_vocab
    return canon_vocab(60000)


class TestRespace:
    def test_recovers_a_simple_classic(self, vocab):
        assert respace("steponnopets", vocab) == "step on no pets".split()

    def test_recovers_the_canonical_example(self, vocab):
        assert respace("amanaplanacanalpanama", vocab) == \
            "a man a plan a canal panama".split()

    def test_output_spells_the_input(self, vocab):
        """The recovered words must use exactly the letters given, in order."""
        for letters in CLASSICS:
            assert "".join(respace(letters, vocab)) == letters

    def test_returns_empty_when_nothing_segments(self, vocab):
        assert respace("qxzjvkw", vocab) == []

    def test_accepts_a_prebuilt_set_without_rebuilding_it(self, vocab):
        """Mining pairs calls this once per attested phrase — 286k times.

        Rebuilding a 60k-word set inside every call made that O(n*m) and the
        run never finished. A caller that already holds the set must be able
        to hand it over.
        """
        prebuilt = frozenset(vocab)
        assert respace("steponnopets", prebuilt) == "step on no pets".split()

    def test_a_set_and_a_list_agree(self, vocab):
        assert respace("nolemonnomelon", vocab) == \
            respace("nolemonnomelon", frozenset(vocab))

    def test_prefers_few_real_words_over_many_short_ones(self, vocab):
        """Without a per-word cost the model spells everything as "a" + "i"."""
        assert respace("panama", vocab) == ["panama"]


class TestRespaceK:
    """Several readings, because the best one is often not the useful one.

    Mining asks whether a mirrored letter run reads as English, and `respace`
    answers with the single most probable segmentation. When a run has an
    attested reading and a slightly more probable unattested one, the pair is
    lost — and lost silently, since a pair was still produced.

    A first attempt to fix this mined the phrase list a second time with the
    halves swapped. That recovered nothing: every entry came back with its own
    flip and no new material, because swapping which half is "given" does not
    change which readings `respace` will produce. k-best does.
    """

    def test_the_first_reading_is_the_one_respace_returns(self, vocab):
        from llm_palindrome.respace import respace_k

        assert respace_k("steponnopets", vocab, k=5)[0] == \
            respace("steponnopets", vocab)

    def test_it_offers_more_than_one_reading(self, vocab):
        from llm_palindrome.respace import respace_k

        # "nopets" has exactly one reading; ambiguity needs a longer run.
        assert len(respace_k("wasitacaroracatisaw", vocab, k=5)) > 1

    def test_every_reading_spells_the_input(self, vocab):
        from llm_palindrome.respace import respace_k

        for reading in respace_k("wasitacaroracatisaw", vocab, k=8):
            assert "".join(reading) == "wasitacaroracatisaw"

    def test_readings_are_ordered_best_first(self, vocab):
        from llm_palindrome.respace import respace_k

        scores = [unigram_score(r)
                  for r in respace_k("nolemonnomelon", vocab, k=6)]
        assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_readings(self, vocab):
        from llm_palindrome.respace import respace_k

        readings = respace_k("steponnopets", vocab, k=10)
        assert len({tuple(r) for r in readings}) == len(readings)

    def test_k_bounds_the_result(self, vocab):
        from llm_palindrome.respace import respace_k

        assert len(respace_k("steponnopets", vocab, k=3)) <= 3

    def test_an_unsegmentable_run_yields_nothing(self, vocab):
        from llm_palindrome.respace import respace_k

        assert respace_k("qxzjvkw", vocab, k=5) == []

    def test_it_finds_an_attested_reading_the_best_one_misses(self, vocab):
        """The case the function exists for, taken from the corpus.

        "a ward" mirrors to "draw a"; read back, the model prefers the single
        word "award" and English attests the two-word reading. Mining needs the
        latter offered even when it is not first.

        Measured, this is rare — k=8 recovers 4 more attested pairs out of 198,
        a 2% gain. An earlier version of this docstring claimed the model's
        favourite is "often" a near-miss; it is not.
        """
        from llm_palindrome.respace import respace_k

        readings = [" ".join(r) for r in respace_k("drawa"[::-1], vocab, k=8)]
        assert readings[0] == "award"
        assert "a ward" in readings


class TestAttestedJoinPreference:
    """A unigram model cannot tell "for ajar" from "for a jar".

    Blind judging of all 61 respaced centres (controls 7/7 and 0/7) accepted
    28. Of the 33 rejections, 14 were not bad palindromes at all but bad
    SEGMENTATIONS — "siri demand i am a maid named iris", "a nut for ajar of
    tuna", "madam in eden i madam", "sit on a potato pa not is". The canon
    sentence was fine; recovery broke it.

    Scoring k-best readings by unigram score plus a bonus per attested join
    fixes some. The weight was swept rather than picked: 0 fixes nothing, 1-4
    fixes three with no regressions, 6+ starts breaking good readings
    ("borrow or rob" becomes "bor row or rob"). Raising k from 24 to 160
    changes nothing, so the remaining failures are not a beam-width problem.

    Three of ten is what this buys. It is recorded as such.
    """

    def test_it_prefers_an_attested_join_over_a_compound(self, vocab):
        from llm_palindrome.respace import respace_attested

        attested = {("for", "a"), ("a", "jar"), ("jar", "of"), ("of", "tuna"),
                    ("a", "nut"), ("nut", "for")}
        got = respace_attested("anutforajaroftuna", vocab, attested)
        assert got == "a nut for a jar of tuna".split()

    def test_it_leaves_a_good_reading_alone(self, vocab):
        """The failure mode of weighting joins too heavily: more words means
        more joins to count, so "panama" splits into "pan am a"."""
        from llm_palindrome.respace import respace_attested

        got = respace_attested("amanaplanacanalpanama", vocab, set())
        assert got == "a man a plan a canal panama".split()

    def test_with_no_attested_pairs_it_matches_plain_respace(self, vocab):
        from llm_palindrome.respace import respace_attested

        assert respace_attested("steponnopets", vocab, set()) == \
            respace("steponnopets", vocab)

    def test_an_unsegmentable_run_yields_nothing(self, vocab):
        from llm_palindrome.respace import respace_attested

        assert respace_attested("qxzjvkw", vocab, set()) == []


class TestUnigramScore:
    def test_a_common_word_beats_a_rare_one(self):
        assert unigram_score(["the"]) > unigram_score(["yak"])

    def test_splitting_a_real_word_is_penalised(self):
        """Otherwise "a" and "i" tile any letter run for free."""
        assert unigram_score(["do", "nut"]) < unigram_score(["donut"])


class TestAgainstTheRecordedSpellings:
    def test_most_classics_come_back_exactly(self, vocab):
        """The bound this is worth shipping at.

        Some are genuinely ambiguous at the letter level — "madaminedenimadam"
        reads as "madam in eden im adam" or "mad amine den im adam" — so exact
        recovery cannot be 10/10 and the failures must be inspected, not
        silently accepted.
        """
        got = {k: " ".join(respace(k, vocab)) for k in CLASSICS}
        exact = [k for k, v in got.items() if v == CLASSICS[k]]
        wrong = {k: (got[k], CLASSICS[k]) for k in CLASSICS if k not in exact}
        assert len(exact) >= 7, wrong

    def test_every_recovery_is_still_a_palindrome(self, vocab):
        """Spacing is invisible to the mirror; recovery must not change that."""
        from llm_palindrome.validator import is_palindrome
        for letters in CLASSICS:
            assert is_palindrome(" ".join(respace(letters, vocab)))


class TestItUnlocksTheAssembler:
    def test_the_respaced_canon_yields_mirror_pairs(self, vocab):
        """The point of the exercise: pairs, where there were none.

        A pair needs a word boundary at the exact letter midpoint. Not every
        palindrome has one — odd letter counts never do — so this asserts the
        canon produces *some*, not many.
        """
        from llm_palindrome.paragraphs import harvest

        canon = json.loads(Path("data/known_palindromes.json").read_text())
        spaced = [" ".join(respace(c, vocab)) for c in canon]
        bank = harvest([s for s in spaced if s])
        assert bank.pairs, "still no pairs — the assembler stays starved"

    def test_the_pairs_read_better_than_the_hunted_ones(self, vocab):
        """Readability is the whole reason to prefer canonical material."""
        from llm_palindrome.paragraphs import harvest

        canon = json.loads(Path("data/known_palindromes.json").read_text())
        spaced = [" ".join(respace(c, vocab)) for c in canon]
        bank = harvest([s for s in spaced if s])
        halves = [w for left, _ in bank.pairs for w in left]
        assert halves
        # "nora aaron" and "bros bacteria" are proper nouns and jargon; the
        # canon's halves are ordinary words.
        mean_len = sum(len(w) for w in halves) / len(halves)
        assert 2.0 <= mean_len <= 6.0, mean_len


class TestSpelledCanon:
    """Stored spellings, because inference cannot win this one.

    Blind judging found 14 of 33 rejected centres were correct palindromes
    wrecked by segmentation. `respace_attested` recovered three. The rest are
    not recoverable by any corpus statistic: choosing "i slam" over "islam"
    needs to know the sentence, and "islam" is a common word that is attested
    beside its neighbours, so unigram, bigram and attested-fraction scoring all
    prefer the wrong reading.

    These are catalogued palindromes with documented spellings. Storing them is
    exact where re-deriving them is a guess. `respace` keeps its job on the
    mining path, where no true spelling exists to store.
    """
    import json as _json
    from pathlib import Path as _Path

    SPELLED = _json.loads(_Path("data/canon_spelled.json").read_text())

    def test_every_stored_spelling_is_a_palindrome(self):
        from llm_palindrome.validator import is_palindrome
        for spelling in self.SPELLED.values():
            assert is_palindrome(spelling), spelling

    def test_each_key_is_the_normalised_form_of_its_value(self):
        from llm_palindrome.validator import normalize
        for letters, spelling in self.SPELLED.items():
            assert normalize(spelling) == letters

    def test_every_entry_is_in_the_novelty_reference(self):
        """A spelling for a palindrome the project does not know is a leak."""
        import json
        from pathlib import Path
        canon = set(json.loads(Path("data/known_palindromes.json").read_text()))
        for letters in self.SPELLED:
            assert letters in canon, letters

    def test_it_fixes_readings_respace_gets_wrong(self):
        from llm_palindrome.respace import canon_vocab, respace
        vocab = canon_vocab(60000)
        wrong = [k for k, v in self.SPELLED.items()
                 if " ".join(respace(k, vocab)) != v]
        assert len(wrong) >= 10, "stored spellings should beat inference"
