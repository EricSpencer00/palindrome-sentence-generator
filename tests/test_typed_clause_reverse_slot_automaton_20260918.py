from experiments.typed_clause_reverse_slot_automaton_20260918 import (
    RIGHT_BUILD_SLOTS,
    LEFT_SLOTS,
    _consume,
    run,
)


def test_right_clause_is_expanded_from_outer_edge():
    assert RIGHT_BUILD_SLOTS == tuple(reversed(LEFT_SLOTS))


def test_consume_carries_cross_phrase_residual():
    # The second comparison consumes a residual from the first phrase rather
    # than requiring a whole phrase pair to have equal length.
    side, residual = _consume("", "", "anaide", "Diana")
    assert (side, residual) == ("L", "e")
    side, residual = _consume(side, residual, "rips", "inspire")
    assert (side, residual) == ("R", "ni")
    side, residual = _consume(side, residual, "nine", "")
    assert (side, residual) == ("L", "ne")


def test_run_records_zero_terminal_closures_without_claiming_readability():
    result = run(max_states=20_000)
    assert result["stats"]["terminal_exact"] == 0
    assert result["stats"]["mechanically_admitted"] == 0
    assert result["reader_gate"].startswith("closed")
