from experiments.working_overhang_coherent_20260921 import JoinScorer
from experiments.working_overhang_growth_20260921 import audit, center_unit
from llm_palindrome.bigram import BigramModel


def test_join_scorer_keeps_directional_seams_distinct():
    model = BigramModel({("red", "royal"): 2}, {"red": 2, "royal": 2})
    scorer = JoinScorer(model, center="now an aide")
    prepended = scorer.word_delta(("royal",), (), "L", "red", "prepend")
    appended = scorer.word_delta((), ("red",), "R", "royal", "append")
    assert prepended != appended


def test_remote_working_candidate_has_independent_exact_audit():
    text = "are macro felt it was noel an era a gas an item smart trams met in a saga arena leon saw title for camera"
    result = audit(text)
    assert result["letters"] == 82
    assert result["two_pointer_exact"] is True
    assert result["sha_equal"] is True
    assert result["validator_exact"] is True
    assert center_unit("Was Noel an era, a gas, an item smart?") == "was noel an era a gas an item smart"
