from experiments.subject_relative_center_chart_20260920 import BINARY, LEXICON, UNARY
from experiments.shared_tape_support_chart_20260920 import chart, normalize


def test_subject_relative_active_control_parses():
    text = "the poet who reads a memo studies a map"
    forest = chart([{c} for c in normalize(text)], LEXICON, BINARY, UNARY)
    assert ("S", 0, len(normalize(text))) in forest


def test_subject_relative_auxiliary_control_parses():
    text = "the poet who has read a poem studies a map"
    forest = chart([{c} for c in normalize(text)], LEXICON, BINARY, UNARY)
    assert ("S", 0, len(normalize(text))) in forest


def test_subject_relative_plural_control_parses():
    text = "some poets who carry two maps study a memo"
    forest = chart([{c} for c in normalize(text)], LEXICON, BINARY, UNARY)
    assert ("S", 0, len(normalize(text))) in forest
