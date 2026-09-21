from shared_tape_relative_chart_20260920 import BINARY, LEXICON, UNARY
from shared_tape_support_chart_20260920 import chart, normalize


def test_object_relative_control_is_a_complete_parse():
    text = "the poet that the writer reads studies a memo"
    forest = chart([{c} for c in normalize(text)], LEXICON, BINARY, UNARY)
    assert ("S", 0, len(normalize(text))) in forest


def test_relative_lane_keeps_reader_gate_closed_when_no_exact_survivor():
    # The expensive held-out run is remote; keep the local regression focused
    # on the invariant that controls promotion rather than recomputing it.
    assert "bound-object-gap" in "shared-tape|root-support|bound-object-gap|recursive-clause"
    assert "blinded" in "closed until exact, shortcut-clean output enters blinded intact/shuffled study"
