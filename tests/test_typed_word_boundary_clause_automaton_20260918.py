from experiments.typed_word_boundary_clause_automaton_20260918 import (
    LEFT_PHASES,
    RIGHT_BUILD_PHASES,
    _consume,
    run,
)


def test_right_build_order_is_reverse_grammar():
    assert LEFT_PHASES[:5] == ("SUBJ_DET", "SUBJ_NOUN", "VERB", "OBJ_DET", "OBJ_NOUN")
    assert RIGHT_BUILD_PHASES[:5] == ("ADJ_NOUN", "ADJ_PREP", "OBJ_NOUN", "OBJ_DET", "VERB")


def test_residual_crosses_multiple_word_boundaries():
    side, debt = _consume("", "", "an", "")
    assert (side, debt) == ("L", "an")
    side, debt = _consume(side, debt, "", "diana")
    assert (side, debt) == ("R", "aid")
    side, debt = _consume(side, debt, "aide", "")
    assert (side, debt) == ("L", "e")
    side, debt = _consume(side, debt, "", "inspire")
    assert (side, debt) == ("R", "ripsni")


def test_seed_is_smoke_control_not_promoted():
    result = run(max_states=20_000, witnesses_per_state=4)
    assert result["stats"]["seed_control_exact"] == 2
    assert result["stats"]["mechanically_admitted"] == 0
    assert result["provenance"]["promoted_seed_scaffold_in_output"] is False
