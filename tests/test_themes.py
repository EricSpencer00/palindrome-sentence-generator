"""Choose centres that share a subject, instead of noticing that some do.

Rung one — a paragraph that is about something — moved once shared-referent
selection was applied to whole sentences rather than to two-word halves. That
selection was done by eye: 8 of the judged centres contain "saw", they were
grouped by hand, and the result beat the same structure with mixed topics under
blind judging.

Eyeballing a frequency table is not a procedure. The same table shows "sir" in
6 centres and "dog" in 5, and no ordering within any cluster was ever chosen.
These tests pin a measured version: score a candidate set by how much content
its sentences share, and search for the best set.

Selection is over CONTENT words only. Function words are shared by everything —
every centre here contains "i" or "a" — so counting them would rank a random
sample as the most cohesive set available.
"""
import pytest

from llm_palindrome.themes import cohesion, content_words, best_cluster

SENTENCES = [
    "was it a cat i saw",
    "was it a rat i saw",
    "was it a car or a cat i saw",
    "lisa bonet ate no basil",
    "ten animals i slam in a net",
    "ma is as selfless as i am",
]


class TestContentWords:
    def test_drops_function_words(self):
        assert content_words("was it a cat i saw") == {"cat", "saw"}

    def test_is_case_insensitive(self):
        assert content_words("Was It A Cat I Saw") == {"cat", "saw"}

    def test_a_sentence_of_only_function_words_is_empty(self):
        assert content_words("it is on the a") == set()

    def test_does_not_count_a_word_twice(self):
        assert content_words("cat cat saw") == {"cat", "saw"}


class TestCohesion:
    def test_two_sentences_sharing_a_word_beat_two_that_do_not(self):
        shared = cohesion(["was it a cat i saw", "was it a rat i saw"])
        apart = cohesion(["was it a cat i saw", "lisa bonet ate no basil"])
        assert shared > apart

    def test_a_single_sentence_has_no_cohesion_to_measure(self):
        assert cohesion(["was it a cat i saw"]) == 0.0

    def test_an_empty_set_is_zero_not_an_error(self):
        assert cohesion([]) == 0.0

    def test_it_is_an_average_not_a_total(self):
        """Otherwise the score grows with set size and every search returns
        the whole inventory."""
        pair = cohesion(["was it a cat i saw", "was it a rat i saw"])
        trio = cohesion(["was it a cat i saw", "was it a rat i saw",
                         "was it a cat i saw"])
        assert trio == pytest.approx(pair, abs=0.34)

    def test_function_words_do_not_create_cohesion(self):
        """Every centre contains "i" or "a"; counting those makes any random
        sample look maximally cohesive."""
        assert cohesion(["it is on a", "the a it is"]) == 0.0


class TestBestCluster:
    def test_finds_the_group_that_shares_a_subject(self):
        got = best_cluster(SENTENCES, size=3)
        assert all("saw" in s for s in got)

    def test_returns_the_requested_size(self):
        assert len(best_cluster(SENTENCES, size=4)) == 4

    def test_asking_for_more_than_exists_returns_everything(self):
        assert len(best_cluster(SENTENCES, size=99)) == len(SENTENCES)

    def test_asking_for_none_returns_nothing(self):
        assert best_cluster(SENTENCES, size=0) == []

    def test_never_repeats_a_sentence(self):
        got = best_cluster(SENTENCES, size=4)
        assert len(set(got)) == len(got)

    def test_it_beats_every_single_word_grouping(self):
        """The bar a greedy walk failed, measured on the real centres.

        Starting from the best PAIR and extending locks onto two near-duplicate
        sentences ("a santa dog lived as a devil god at nasa" beside "a santa
        lived as a devil at nasa" share five words) and then dilutes: cohesion
        0.476 at size 7, against 1.095 for the sentences sharing "saw". A
        strong pair plus weak additions loses to consistent moderate overlap.

        Grouping by each content word in turn is exhaustive over one-word
        themes, so it can never do worse than the best of them.
        """
        pool = SENTENCES + ["a santa lived as a devil at nasa",
                            "a santa dog lived as a devil god at nasa"]
        by_word = []
        for word in {w for s in pool for w in content_words(s)}:
            group = [s for s in pool if word in content_words(s)]
            if len(group) >= 3:
                by_word.append(cohesion(group[:3]))
        assert cohesion(best_cluster(pool, size=3)) >= max(by_word)


class TestRefrainOrder:
    """Where a sentence sits in a refrain is not arbitrary.

    A refrain reads A B C D C B A: outward from the centre in both directions,
    and the centre is the only sentence that does not repeat. Three orderings
    of the same seven sentences were judged blind against real-prose and salad
    controls (`runs/order_key.json`):

      questions outward, declaration at the centre   ranked 2nd (best of three)
      declarations outward, question at the centre   ranked 3rd
      questions and declarations interleaved         ranked 4th

    The judge's reason was structural: the strongest line belongs on the turn,
    and the outer positions should carry the least resolved material. So the
    rule is settled by the shape, not by taste — the centre is the one seat
    that is heard once.
    """

    QUESTIONS = ["was it a cat i saw", "was it a rat i saw"]
    STATEMENTS = ["delia saw i was ailed", "able was i ere i saw elba"]

    def test_a_question_is_recognised(self):
        from llm_palindrome.themes import is_question

        assert is_question("was it a cat i saw")
        assert is_question("are we not drawn onward to new era")
        assert is_question("can i see bees in a cave")

    def test_a_statement_is_not(self):
        from llm_palindrome.themes import is_question

        assert not is_question("delia saw i was ailed")
        assert not is_question("able was i ere i saw elba")

    def test_statements_land_nearest_the_centre(self):
        from llm_palindrome.themes import order_for_refrain

        got = order_for_refrain(self.QUESTIONS + self.STATEMENTS)
        assert not from_question(got[-1]), got

    def test_questions_land_outermost(self):
        from llm_palindrome.themes import is_question, order_for_refrain

        got = order_for_refrain(self.QUESTIONS + self.STATEMENTS)
        assert is_question(got[0]), got

    def test_it_keeps_every_sentence(self):
        from llm_palindrome.themes import order_for_refrain

        got = order_for_refrain(self.QUESTIONS + self.STATEMENTS)
        assert sorted(got) == sorted(self.QUESTIONS + self.STATEMENTS)

    def test_all_statements_is_not_an_error(self):
        from llm_palindrome.themes import order_for_refrain

        assert sorted(order_for_refrain(self.STATEMENTS)) == \
            sorted(self.STATEMENTS)

    def test_empty_is_empty(self):
        from llm_palindrome.themes import order_for_refrain

        assert order_for_refrain([]) == []


def from_question(sentence):
    from llm_palindrome.themes import is_question
    return is_question(sentence)


class TestTrimToTheme:
    """Return a shorter paragraph rather than pad a thin theme.

    A prompt matching two centres used to get five strangers to fill the
    requested length. Judged blind, that is worse than simply returning the
    two: a 5-sentence refrain that stays on subject ranked ABOVE a 13-sentence
    one that opens on the same subject and collapses into Lisa Bonet, animals
    in a net and Stella (`runs` — the padded/short batch). Cohesion 1.333
    against 0.238, and here the metric and the judge agree.

    Padding is worse than brevity because a reader reads the whole thing: the
    strangers do not add to the subject, they remove it.
    """

    CORE = ["now sir a war is won", "red rum sir is murder",
            "no sir away a papaya war is on"]
    STRANGERS = ["lisa bonet ate no basil", "stella won no wallets"]

    def test_it_drops_a_sentence_sharing_nothing(self):
        from llm_palindrome.themes import trim_to_theme

        got = trim_to_theme(self.CORE + self.STRANGERS)
        assert sorted(got) == sorted(self.CORE)

    def test_it_keeps_a_coherent_set_whole(self):
        from llm_palindrome.themes import trim_to_theme

        assert sorted(trim_to_theme(self.CORE)) == sorted(self.CORE)

    def test_it_preserves_order(self):
        from llm_palindrome.themes import trim_to_theme

        got = trim_to_theme([self.CORE[0], self.STRANGERS[0], self.CORE[1]])
        assert got == [self.CORE[0], self.CORE[1]]

    def test_it_never_returns_nothing(self):
        """A set with no shared content at all still has to answer."""
        from llm_palindrome.themes import trim_to_theme

        got = trim_to_theme(self.STRANGERS)
        assert len(got) >= 1

    def test_a_single_sentence_survives(self):
        from llm_palindrome.themes import trim_to_theme

        assert trim_to_theme(self.CORE[:1]) == self.CORE[:1]

    def test_empty_is_empty(self):
        from llm_palindrome.themes import trim_to_theme

        assert trim_to_theme([]) == []

    def test_trimming_raises_cohesion(self):
        from llm_palindrome.themes import cohesion, trim_to_theme

        padded = self.CORE + self.STRANGERS
        assert cohesion(trim_to_theme(padded)) > cohesion(padded)


class TestQuestionClassification:
    """Audited against all 49 shipped centres, which found three defects.

    `is_question` drives the seating rule — questions outward, the firmest
    statement on the turn — off a list of openers matched at position 0. The
    inventory has outgrown that:

      "eva can i see bees in a cave"   a question whose inversion follows a
      "wont i panic in a pit now"      vocative, so it never matched
      "may a moody baby doom a yam"    an optative wish, matched as a question

    A question is auxiliary-subject inversion, and English lets a vocative or
    an adverb precede it. So the openers are looked for in the first few words
    rather than only at the start — and "may a", which is not an inversion at
    all, comes out.
    """

    QUESTIONS = ["was it a cat i saw", "eva can i see bees in a cave",
                 "wont i panic in a pit now",
                 "are we not drawn onward we few drawn onward to new era"]
    NOT_QUESTIONS = ["may a moody baby doom a yam", "delia saw i was ailed",
                     "able was i ere i saw elba", "poor dan is in a droop",
                     "now i see bees i won"]

    def test_every_question_in_the_inventory_is_found(self):
        from llm_palindrome.themes import is_question

        for sentence in self.QUESTIONS:
            assert is_question(sentence), sentence

    def test_nothing_else_is_called_one(self):
        from llm_palindrome.themes import is_question

        for sentence in self.NOT_QUESTIONS:
            assert not is_question(sentence), sentence

    def test_an_inversion_after_a_vocative_still_counts(self):
        from llm_palindrome.themes import is_question

        assert is_question("eva can i see bees in a cave")

    def test_a_wish_is_not_a_question(self):
        """"May a moody baby doom a yam" is optative, not interrogative."""
        from llm_palindrome.themes import is_question

        assert not is_question("may a moody baby doom a yam")

    def test_the_opener_is_not_hunted_arbitrarily_far_in(self):
        """Otherwise any sentence containing "is it" anywhere is a question."""
        from llm_palindrome.themes import is_question

        assert not is_question(
            "a long declarative sentence that happens to say was it later")
