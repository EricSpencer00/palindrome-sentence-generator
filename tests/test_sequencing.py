"""Order the mirror-pairs so the paragraph reads, in both directions at once.

The canon now yields 23 readable pairs ("lived on decaf" / "faced no devil"),
so the assembler is no longer starved. What it produces is still a list: the
halves are English, their sequence is not. Nesting

    L1 L2 ... Lk  CENTER  Rk ... R2 R1

makes L_i adjacent to L_{i+1} on the way in, and R_{i+1} adjacent to R_i on the
way out. Those are the same choice. Improving a junction on the left fixes the
corresponding junction on the right, and there is no ordering that optimises
one side alone — this is the mirror cost, reappearing at the level of sequence
rather than letters.

That makes it an ordering problem over a small set, not a wall. These tests pin
a greedy sequencer whose objective scores both junctions a placement creates.
"""
import pytest

from llm_palindrome.sequencing import junction_cost, order_pairs


class FakeBigrams:
    """Only "decaf faced" and "time emit" are fluent; everything else is not."""

    GOOD = {("on", "live"), ("decaf", "faced"), ("time", "emit")}

    def forward(self, prev, word):
        return -1.0 if (prev, word) in self.GOOD else -8.0

    def backward(self, word, nxt):
        return self.forward(word, nxt)


PAIRS = [
    (["lived", "on", "decaf"], ["faced", "no", "devil"]),
    (["live", "on", "time"], ["emit", "no", "evil"]),
    (["evil", "rats", "on"], ["no", "star", "live"]),
]


class TestJunctionCost:
    def test_scores_the_left_seam_and_the_right_seam(self, ):
        """A placement creates two adjacencies, not one."""
        bg = FakeBigrams()
        a, b = PAIRS[0], PAIRS[1]
        cost = junction_cost(a, b, bg)
        left = bg.forward("decaf", "live")
        right = bg.forward("emit", "faced")
        assert cost == pytest.approx(left + right)

    def test_a_fluent_seam_scores_higher_than_a_broken_one(self):
        bg = FakeBigrams()
        good = junction_cost(PAIRS[2], PAIRS[1], bg)   # ...on | live...
        bad = junction_cost(PAIRS[1], PAIRS[2], bg)    # ...time | evil...
        assert good > bad

    def test_it_cannot_be_improved_on_one_side_only(self):
        """The property that makes this the mirror cost and not a free lunch.

        Swapping two pairs changes the left seam and the right seam together;
        no ordering exists that keeps the good left seam and drops the bad
        right one.
        """
        bg = FakeBigrams()
        a, b = PAIRS[0], PAIRS[1]
        forward_left = bg.forward(a[0][-1], b[0][0])
        forward_right = bg.forward(b[1][-1], a[1][0])
        assert junction_cost(a, b, bg) == forward_left + forward_right


class TestOrderPairs:
    def test_uses_every_pair_exactly_once(self):
        out = order_pairs(PAIRS, FakeBigrams())
        assert sorted(map(str, out)) == sorted(map(str, PAIRS))

    def test_puts_the_fluent_seam_together(self):
        """"evil rats on" then "live on time" — the only good join available."""
        out = order_pairs(PAIRS, FakeBigrams())
        seq = [" ".join(l) for l, _ in out]
        assert seq.index("evil rats on") + 1 == seq.index("live on time")

    def test_handles_a_single_pair(self):
        assert order_pairs(PAIRS[:1], FakeBigrams()) == PAIRS[:1]

    def test_handles_no_pairs(self):
        assert order_pairs([], FakeBigrams()) == []

    def test_the_result_still_assembles_to_a_palindrome(self):
        """Ordering must never be able to break the invariant."""
        from llm_palindrome.paragraphs import assemble
        from llm_palindrome.validator import is_palindrome

        out = order_pairs(PAIRS, FakeBigrams())
        assert is_palindrome(" ".join(assemble(out, None)))


class TestCompose:
    """Choose WHICH units as well as their order, scoring the finished text.

    `order_pairs` takes the units it is given and optimises seams. That leaves
    the paragraph with no subject, because seam scores are local: "roll a" and
    "six of" join as smoothly as "war as a" and "raw food", and only the second
    pair is about anything.

    Selection is where a through-line can come from, and with 26 usable units
    the subsets are searchable. The objective is the rendered paragraph itself
    rather than a proxy — the README records that every proxy tried failed
    against judge verdicts, so the scorer is injected and the caller decides
    what "reads well" means.
    """

    def score_by_shared_words(self, text):
        """Rewards a paragraph whose units share vocabulary — a stand-in
        topic model, so the test does not need a language model."""
        words = [w for w in text.lower().replace(".", "").split()]
        return sum(n - 1 for n in
                   {w: words.count(w) for w in set(words)}.values())

    def test_picks_the_requested_number_of_units(self):
        from llm_palindrome.sequencing import compose

        out = compose(PAIRS, self.score_by_shared_words, want=2)
        assert len(out) == 2

    def test_never_repeats_a_unit(self):
        from llm_palindrome.sequencing import compose

        out = compose(PAIRS, self.score_by_shared_words, want=3)
        assert len({" ".join(l) for l, _ in out}) == 3

    def test_wanting_more_than_exists_returns_everything(self):
        from llm_palindrome.sequencing import compose

        out = compose(PAIRS, self.score_by_shared_words, want=99)
        assert len(out) == len(PAIRS)

    def test_wanting_none_returns_nothing(self):
        from llm_palindrome.sequencing import compose

        assert compose(PAIRS, self.score_by_shared_words, want=0) == []

    def test_the_result_is_still_a_palindrome(self):
        from llm_palindrome.paragraphs import assemble
        from llm_palindrome.validator import is_palindrome

        from llm_palindrome.sequencing import compose

        out = compose(PAIRS, self.score_by_shared_words, want=3)
        assert is_palindrome(" ".join(assemble(out, None)))

    def test_it_prefers_the_higher_scoring_selection(self):
        """Given a scorer that only likes one unit, that unit must be chosen."""
        from llm_palindrome.sequencing import compose

        def likes_rats(text):
            # By WORD, and a word only ONE pair has: "evil" appears in both
            # "Emit no evil" and "Evil rats on", and as a substring of "devil".
            return 10.0 if "rats" in text.lower().replace(".", "").split() \
                else 0.0

        out = compose(PAIRS, likes_rats, want=1)
        assert " ".join(out[0][0]) == "evil rats on"

    def test_the_scorer_sees_the_rendered_paragraph(self):
        """Not a word list, not the halves — the text a reader would get."""
        from llm_palindrome.sequencing import compose

        seen = []

        def spy(text):
            seen.append(text)
            return 0.0

        compose(PAIRS, spy, want=1)
        assert seen and all(isinstance(t, str) and "." in t for t in seen)


class TestRepetitionRate:
    """The quantity that exposed GPT-2 gaming the selection.

    Optimising the LM score over the unit pool improved it from -2.80 to -2.29
    and moved this number from 0.356 to 0.471: the search was buying "a" twelve
    times, not a subject. Any objective used for selection has to be checked
    against it, so it is measured code rather than a note.
    """

    def test_no_repeats_scores_zero(self):
        from llm_palindrome.sequencing import repetition_rate

        assert repetition_rate("Step on. No pets.") == 0.0

    def test_counts_repeats_as_a_share_of_words(self):
        from llm_palindrome.sequencing import repetition_rate

        # four words, "a" appears twice -> one repeat in four
        assert repetition_rate("A time. A dot.") == pytest.approx(0.25)

    def test_is_case_and_punctuation_insensitive(self):
        from llm_palindrome.sequencing import repetition_rate

        assert repetition_rate("A time. a TIME.") == \
            repetition_rate("a time a time")

    def test_empty_text_is_zero_not_an_error(self):
        from llm_palindrome.sequencing import repetition_rate

        assert repetition_rate("") == 0.0

    def test_a_guarded_scorer_rejects_a_repetitive_candidate(self):
        from llm_palindrome.sequencing import guarded

        scorer = guarded(lambda t: 1.0, max_rate=0.3)
        assert scorer("Step on. No pets.") == 1.0
        assert scorer("A a a a a b.") == float("-inf")


class TestCadenceConcentration:
    """The gaming mode that appeared once word repetition was guarded.

    With `repetition_rate` bounded, optimising GPT-2 over 218 units returned
    "Partner is. Sign is. Warning is. Flower is. Garden is. Side it. Tied it.
    Stole it." — no repeated word beyond the bound, and no subject either. The
    repeated element is the SHAPE: 8 of 28 units ended in "is" (0.29) against
    2 of 14 for the unoptimised baseline (0.14).

    Each guard closes one door and the search finds the next, which is the
    argument against trusting any fluency proxy on an inventory this thin.
    """

    def test_varied_endings_score_low(self):
        from llm_palindrome.sequencing import cadence_concentration

        assert cadence_concentration("Step on. No pets. A time.") < 0.5

    def test_a_single_repeated_ending_scores_high(self):
        from llm_palindrome.sequencing import cadence_concentration

        text = "Partner is. Sign is. Warning is. Flower is."
        assert cadence_concentration(text) == 1.0

    def test_empty_text_is_zero_not_an_error(self):
        from llm_palindrome.sequencing import cadence_concentration

        assert cadence_concentration("") == 0.0

    def test_the_guard_bounds_it_too(self):
        from llm_palindrome.sequencing import guarded

        scorer = guarded(lambda t: 1.0, max_rate=0.9, max_cadence=0.5)
        assert scorer("Step on. No pets. A time.") == 1.0
        assert scorer("Sign is. Flower is. Garden is.") == float("-inf")


class TestAgainstTheRealCanon:
    @pytest.mark.slow
    def test_ordering_beats_the_arbitrary_order(self):
        """Measured on the material this actually runs against."""
        import json
        from pathlib import Path

        from llm_palindrome.bigram import BigramModel
        from llm_palindrome.paragraphs import assemble, harvest
        from llm_palindrome.respace import canon_vocab, respace

        vocab = canon_vocab(60000)
        canon = json.loads(Path("data/known_palindromes.json").read_text())
        bank = harvest([" ".join(respace(c, vocab)) for c in canon])
        bg = BigramModel.from_file("data/count_2w.txt")

        def total(pairs):
            words = assemble(pairs, None)
            return sum(bg.forward(a, b) for a, b in zip(words, words[1:]))

        assert total(order_pairs(bank.pairs, bg)) > total(bank.pairs)
