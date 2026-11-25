"""The spectrum verifier puts salad at one end, stock phrases at the other,
and readable, surprising palindromes in between."""
import os
import time

import pytest

from llm_palindrome import spectrum

pytestmark = pytest.mark.skipif(not os.path.exists(spectrum.BIGRAMS),
                                reason="needs data/count_2w.txt and the NLTK Brown corpus")


@pytest.fixture(scope="module")
def sp():
    return spectrum.Spectrum()


def test_salad_sits_at_the_garbage_end(sp):
    a = sp.assess("race cars are here we most are here may be many men or off")
    assert a["position"] < 0.2 and a["score"] < 0.1


def test_classics_sit_at_the_ism_end(sp):
    for text in ("Step on no pets.", "A man, a plan, a canal: Panama."):
        a = sp.assess(text)
        assert a["position"] > 0.6 and a["score"] < 0.1, text


def test_edgy_readable_text_scores_highest(sp):
    middle = sp.assess("Live deviant, naive devil!")["score"]
    assert middle > sp.assess("Step on no pets.")["score"]
    assert middle > sp.assess("devil naive live deviant")["score"]


def test_repeated_content_and_lowercase_names_are_garbage(sp):
    assert sp.assess("Not as it is a level as it is a ton.")["score"] == 0
    assert sp.assess("Pitt ahmet sara nara stem hat tip.")["readable"] < 0.3


def test_fast_enough_for_the_search_loop(sp):
    texts = ["Pale roses use sore lap.", "Burst its rub.", "Sew olive donor pro no devil owes."] * 100
    t = time.perf_counter()
    for x in texts:
        sp.assess(x)
    assert (time.perf_counter() - t) / len(texts) < 0.002
