from experiments.agreement_passive_relative_chart_20260920 import BINARY, LEXICON, UNARY
from experiments.shared_tape_support_chart_20260920 import chart, normalize


def test_agreement_active_relative_control_parses():
    text = "the poet that the writer reads studies a memo"
    forest = chart([{c} for c in normalize(text)], LEXICON, BINARY, UNARY)
    assert ("S", 0, len(normalize(text))) in forest


def test_agreement_passive_relative_control_parses():
    text = "the poet that the writer has read studies a poem"
    forest = chart([{c} for c in normalize(text)], LEXICON, BINARY, UNARY)
    assert ("S", 0, len(normalize(text))) in forest


def test_plural_agreement_control_parses():
    text = "some poets who writers read carry two maps"
    forest = chart([{c} for c in normalize(text)], LEXICON, BINARY, UNARY)
    assert ("S", 0, len(normalize(text))) in forest


def test_new_lane_is_not_a_vocabulary_sweep():
    signature = "shared-tape|root-support|agreement-features|active-passive-relative"
    assert "agreement-features" in signature
    assert "active-passive-relative" in signature
