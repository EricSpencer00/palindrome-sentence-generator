"""Tests for assembling paragraph-length palindromes from short units.

The mirror costs 3.3 bits per free letter, so prose cannot survive at
paragraph length — but a paragraph does not have to BE prose end to end. If
every unit is either itself a palindrome or half of a mirror-pair, then

    L1 L2 ... Lk  CENTER  Rk ... R2 R1

is a palindrome by construction at any length: each Li's letters reversed are
its partner Ri's, and the center reads the same both ways. The constraint is
paid inside units short enough to hunt exhaustively, and choosing WHICH units
— the only place coherence can come from — carries no letter constraint at all.

The harvest side: every palindrome the hunt found whose mirror point lands on
a word boundary IS a mirror-pair, its two halves already readable, because the
whole thing was scored as readable.
"""
from llm_palindrome.paragraphs import assemble, harvest
from llm_palindrome.validator import is_palindrome, normalize


class TestHarvest:
    def test_a_palindrome_splitting_on_a_word_boundary_yields_a_pair(self):
        got = harvest(["rob a lot to labor"])
        assert got.pairs == [(["rob", "a", "lot"], ["to", "labor"])]

    def test_the_two_halves_mirror_each_other(self):
        (left, right), = harvest(["rob a lot to labor"]).pairs
        assert normalize(" ".join(left)) == normalize(" ".join(right))[::-1]

    def test_a_palindrome_with_a_central_word_is_a_center_not_a_pair(self):
        got = harvest(["step on no pets"])          # even split: pair
        assert got.pairs and not got.centers
        got = harvest(["rats live on no evil star"])  # 20 letters, splits clean
        assert got.pairs

    def test_odd_centered_palindrome_is_kept_as_a_center(self):
        got = harvest(["never odd or even"])   # mirror falls inside "odd"
        assert got.centers == [["never", "odd", "or", "even"]]

    def test_non_palindromes_are_rejected(self):
        got = harvest(["this is not one"])
        assert not got.pairs and not got.centers


class TestAssemble:
    PAIRS = [(["rob", "a", "lot"], ["to", "labor"]),
             (["step", "on"], ["no", "pets"])]
    CENTER = ["never", "odd", "or", "even"]

    def test_the_assembly_is_a_palindrome(self):
        words = assemble(self.PAIRS, self.CENTER)
        assert is_palindrome(" ".join(words))

    def test_it_is_a_palindrome_with_no_center_too(self):
        words = assemble(self.PAIRS, None)
        assert is_palindrome(" ".join(words))

    def test_rights_nest_in_reverse_order_of_lefts(self):
        words = assemble(self.PAIRS, self.CENTER)
        text = " ".join(words)
        assert text.index("rob") < text.index("step")
        assert text.index("no pets") < text.index("to labor")

    def test_any_subset_and_order_of_pairs_still_assembles_a_palindrome(self):
        words = assemble([self.PAIRS[1], self.PAIRS[0]], self.CENTER)
        assert is_palindrome(" ".join(words))

    def test_empty_input_is_empty(self):
        assert assemble([], None) == []

    def test_length_grows_without_breaking_the_mirror(self):
        words = assemble(self.PAIRS * 6, self.CENTER)
        assert len(normalize(" ".join(words))) > 150
        assert is_palindrome(" ".join(words))


class TestRender:
    """Each unit half is its own sentence: the classics have always spent
    punctuation freely, and normalize() makes it free here too."""

    def test_each_half_renders_as_a_sentence(self):
        from llm_palindrome.paragraphs import render
        out = render([(["rob", "a", "lot"], ["to", "labor"])], None)
        assert out == "Rob a lot. To labor."

    def test_the_letters_are_untouched(self):
        from llm_palindrome.paragraphs import render
        pairs = [(["rob", "a", "lot"], ["to", "labor"]),
                 (["step", "on"], ["no", "pets"])]
        out = render(pairs, ["never", "odd", "or", "even"])
        assert normalize(out) == normalize(" ".join(
            ["rob","a","lot","step","on","never","odd","or","even",
             "no","pets","to","labor"]))

    def test_rendered_text_is_still_a_palindrome(self):
        from llm_palindrome.paragraphs import render
        out = render([(["step", "on"], ["no", "pets"])], None)
        assert is_palindrome(out)


class TestFamilyCap:
    """Diversity has to be enforced, not hoped for.

    The v3 paragraph failed its judge at 8/24 because selection drew five
    pairs from the "dial an if" family: the bank's usable pairs clustered in
    two families, and a menu that shows near-duplicates gets near-duplicate
    paragraphs. A pair's FAMILY is its mirror-core signature — the words its
    two halves share with every sibling — approximated by the closing half's
    first two words.
    """

    def test_caps_pairs_per_family(self):
        from llm_palindrome.paragraphs import diversify
        pairs = [(["a", "x"], ["no", "it", "one"]),
                 (["b", "y"], ["no", "it", "two"]),
                 (["c", "z"], ["no", "it", "three"]),
                 (["d", "w"], ["dial", "an", "if"])]
        out = diversify(pairs, per_family=2)
        assert len(out) == 3
        assert (["d", "w"], ["dial", "an", "if"]) in out

    def test_keeps_input_order_within_the_cap(self):
        from llm_palindrome.paragraphs import diversify
        pairs = [(["a"], ["no", "it", "one"]), (["b"], ["no", "it", "two"])]
        assert diversify(pairs, per_family=1) == [pairs[0]]

    def test_cap_of_zero_is_empty(self):
        from llm_palindrome.paragraphs import diversify
        assert diversify([(["a"], ["no", "it"])], per_family=0) == []

    def test_a_pair_and_its_flip_are_the_same_material(self):
        """When both halves are attested, mining finds the pair both ways.

        "not as || sat on" and "sat on || not as" are both mined, because each
        half is an attested phrase that mirrors to the other. Using both puts
        the identical four words in one paragraph twice, which is what made a
        20-unit selection render "Not as ... Sat on ... Not as ... Sat on".
        """
        from llm_palindrome.paragraphs import diversify
        pairs = [(["not", "as"], ["sat", "on"]),
                 (["sat", "on"], ["not", "as"]),
                 (["six", "of"], ["fox", "is"])]
        out = diversify(pairs, per_family=3)
        assert len(out) == 2
        assert out[0] == pairs[0]

    def test_caps_families_that_cluster_on_the_OPENING_half(self):
        """Mined pairs cluster on the left, harvested ones on the right.

        Harvested pairs share a mirror-core, so siblings look alike in their
        closing half. Mined pairs are keyed by an attested opening phrase, so
        "sites may", "sites but", "sites not" and "sites was" are siblings with
        four different closing halves — and a closing-half key let all four
        into one paragraph.
        """
        from llm_palindrome.paragraphs import diversify
        pairs = [(["sites", "may"], ["yam", "set", "is"]),
                 (["sites", "but"], ["tub", "set", "is"]),
                 (["sites", "not"], ["ton", "set", "is"]),
                 (["notes", "are"], ["era", "set", "on"])]
        out = diversify(pairs, per_family=2)
        assert len(out) == 3
        assert (["notes", "are"], ["era", "set", "on"]) in out


class TestRefrainForm:
    """A mirrored sequence of self-palindromic sentences.

    The pair route died on data: judged strictly, zero of the bank's top pairs
    had two readable halves — a mirror half reads or its partner does, not
    both. But the hunt's whole short palindromes pass judges as single
    sentences, and a SEQUENCE arranged A B C D C B A of self-palindromic units
    is a palindrome by construction. The repeats are refrains: the form every
    long human palindromic poem already uses.
    """

    def test_refrain_of_palindromic_sentences_is_a_palindrome(self):
        from llm_palindrome.paragraphs import refrain
        units = ["step on no pets", "no it call action", "never odd or even"]
        out = refrain(units)
        assert is_palindrome(" ".join(out))

    def test_each_unit_appears_at_most_twice(self):
        from llm_palindrome.paragraphs import refrain
        out = refrain(["step on no pets", "no it call action", "was it a rat i saw"])
        joined = " | ".join([" ".join(out)])
        assert joined.count("step on no pets") == 2
        assert joined.count("was it a rat i saw") == 1   # the center

    def test_rejects_a_unit_that_is_not_a_palindrome(self):
        from llm_palindrome.paragraphs import refrain
        import pytest as _pytest
        with _pytest.raises(ValueError):
            refrain(["this is not one", "step on no pets"])

    def test_single_unit_is_itself(self):
        from llm_palindrome.paragraphs import refrain
        assert refrain(["step on no pets"]) == ["step on no pets"]


class TestMultiSentenceCenter:
    """A refrain core is many sentences, not one.

    `render` took the center as a flat word list and joined it into a single
    sentence, so a 7-unit refrain came out as one 200-letter run-on. The center
    has to keep its own sentence boundaries.
    """

    def test_center_sentences_render_separately(self):
        """Two centre units, each a palindrome, so their join still mirrors.

        This test previously used two DIFFERENT centres and asserted the
        resulting string — which was not a palindrome. It was encoding the
        missing validation as expected behaviour.
        """
        from llm_palindrome.paragraphs import render
        out = render([(["step", "on"], ["no", "pets"])],
                     center_units=["never odd or even", "never odd or even"])
        assert out == ("Step on. Never odd or even. Never odd or even. No pets.")
        assert is_palindrome(out)

    def test_letters_preserved_with_center_units(self):
        from llm_palindrome.paragraphs import render
        out = render([(["step", "on"], ["no", "pets"])],
                     center_units=["never odd or even"])
        assert normalize(out) == normalize("step on never odd or even no pets")

    def test_flat_center_still_works(self):
        from llm_palindrome.paragraphs import render
        out = render([(["step", "on"], ["no", "pets"])], ["never", "odd", "or", "even"])
        assert out == "Step on. Never odd or even. No pets."


class TestNovelty:
    """A find has to be checked against the palindrome RECORD, not our corpora.

    "No devil lived on" was reported as novel because it appears in neither
    Brown nor WikiText — true, and irrelevant: it is a catalogued classic. The
    corpora were never where a known palindrome would live.
    """

    def test_a_known_classic_is_not_novel(self):
        from llm_palindrome.paragraphs import is_novel_palindrome
        assert not is_novel_palindrome("no devil lived on")
        assert not is_novel_palindrome("step on no pets")

    def test_punctuation_and_case_do_not_hide_a_classic(self):
        from llm_palindrome.paragraphs import is_novel_palindrome
        assert not is_novel_palindrome("No, devil lived on!")

    def test_an_unknown_palindrome_is_novel(self):
        from llm_palindrome.paragraphs import is_novel_palindrome
        assert is_novel_palindrome("non academia aimed a canon")


class TestWordLevelPalindrome:
    """Word-order palindromes are a different constraint entirely.

    "Fall leaves as soon as leaves fall" reads the same by WORD order and not
    by letter — popular lists conflate the two, and `is_palindrome` correctly
    rejects it. The distinction matters because a word-order palindrome pays
    no per-letter mirror cost: it can be arbitrarily long and still read.
    """

    def test_word_palindrome_accepted(self):
        from llm_palindrome.paragraphs import is_word_palindrome
        assert is_word_palindrome("fall leaves as soon as leaves fall")

    def test_letter_palindrome_is_not_automatically_a_word_one(self):
        from llm_palindrome.paragraphs import is_word_palindrome
        assert not is_word_palindrome("step on no pets")

    def test_ignores_case_and_punctuation(self):
        from llm_palindrome.paragraphs import is_word_palindrome
        assert is_word_palindrome("King, are you glad you are king!")

    def test_rejects_ordinary_prose(self):
        from llm_palindrome.paragraphs import is_word_palindrome
        assert not is_word_palindrome("the ice held until march")


class TestWordAssembly:
    """Nest WHOLE sentences, not half-sentences.

    Splitting each unit at its midpoint made the paragraph's outer lines
    fragments — "Fishermen mended." from "Fishermen mended nets, mended
    fishermen." A word-palindromic sentence and its own word-reversal are the
    bracket pair; the sentence stays intact and the reversal is a real line.
    """

    def test_a_sentence_and_its_reversal_bracket_a_center(self):
        from llm_palindrome.paragraphs import word_assemble, is_word_palindrome
        out = word_assemble(["boats crossed water"], "mist rose slowly rose mist")
        assert is_word_palindrome(out)

    def test_outer_lines_are_whole_sentences(self):
        from llm_palindrome.paragraphs import word_assemble
        out = word_assemble(["boats crossed water"], "mist rose mist")
        assert out.startswith("Boats crossed water.")
        assert out.endswith("Water crossed boats.")

    def test_nesting_order_reverses(self):
        from llm_palindrome.paragraphs import word_assemble, is_word_palindrome
        out = word_assemble(["boats crossed water", "geese left lake"],
                            "mist rose mist")
        assert is_word_palindrome(out)
        assert out.index("Boats") < out.index("Geese")
        assert out.index("Lake left geese") < out.index("Water crossed boats")

    def test_center_must_be_word_palindromic(self):
        from llm_palindrome.paragraphs import word_assemble
        import pytest as _p
        with _p.raises(ValueError):
            word_assemble(["boats crossed water"], "the ice held until march")


class TestRenderValidates:
    """`assemble` asserts the mirror; `render` did not, and silently produced a
    non-palindrome when handed two centre units — a palindrome has one centre,
    and two only work if their concatenation is itself palindromic."""

    def test_two_arbitrary_centres_are_rejected(self):
        from llm_palindrome.paragraphs import render
        import pytest as _p
        with _p.raises(AssertionError):
            render([(["step", "on"], ["no", "pets"])],
                   center_units=["do geese see god", "never odd or even"])

    def test_one_centre_is_fine(self):
        from llm_palindrome.paragraphs import render
        out = render([(["step", "on"], ["no", "pets"])],
                     center_units=["never odd or even"])
        assert is_palindrome(out)

    def test_two_centres_whose_join_mirrors_are_accepted(self):
        from llm_palindrome.paragraphs import render
        out = render([(["step", "on"], ["no", "pets"])],
                     center_units=["never odd or even", "never odd or even"])
        assert is_palindrome(out)
