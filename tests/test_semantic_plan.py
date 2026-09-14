import json
from pathlib import Path

from llm_palindrome.semantic_plan import SemanticPlan, render_plan_prompt


def plan():
    return SemanticPlan.from_dict({
        "intent": "A careful baker checks the oven before serving bread.",
        "facts": [
            {"fact_id": "f1", "subject": "baker", "predicate": "checks", "object": "oven", "terms": ["baker", "checks", "oven"]},
            {"fact_id": "f2", "subject": "baker", "predicate": "serves", "object": "bread", "terms": ["baker", "serves", "bread"]},
        ],
        "links": ["f1 precedes f2"],
    })


def test_plan_is_frozen_and_hashable():
    value = plan()
    assert len(value.sha256) == 64
    assert value.as_dict()["facts"][0]["fact_id"] == "f1"


def test_coverage_is_diagnostic_and_tracks_complete_fact_terms():
    value = plan()
    assert value.coverage_vector(("the", "baker", "checks", "the", "oven")) == (True, False)
    assert value.minimum_coverage(("baker", "checks", "oven", "baker", "serves", "bread")) == 2
    phrase_plan = SemanticPlan.from_dict({
        "intent": "A team presents findings.",
        "facts": [
            {"fact_id": "f1", "subject": "e1", "predicate": "presents", "object": "e2", "terms": ["research team", "presents", "findings"]},
            {"fact_id": "f2", "subject": "e1", "predicate": "meets", "object": "e3", "terms": ["research team", "meets", "panel"]},
        ],
        "links": [],
    })
    assert phrase_plan.minimum_coverage(("the", "research", "team", "presents", "findings")) == 1


def test_plan_schema_rejects_duplicate_or_wrong_fact_counts():
    raw = plan().as_dict()
    raw["facts"] = [raw["facts"][0]]
    try:
        SemanticPlan.from_dict(raw)
    except ValueError as error:
        assert str(error) == "semantic_plan_requires_two_to_four_distinct_facts"
    else:
        raise AssertionError("expected schema rejection")


def test_plan_prompt_does_not_ask_for_palindrome_text():
    assert "palindrome" not in render_plan_prompt().lower()


def test_plan_conditioned_driver_audits_exactness_and_reader_gate():
    from experiments.plan_conditioned_exact_search_20260913 import plan_candidate_audit

    value = plan()
    audit = plan_candidate_audit("A dog, god a.", value, seed=3)
    assert audit["exact_editor_audit"]["independent_exactness"]["direct_symmetric_position_comparison"]
    assert audit["human_reader_study"] == "not_run"
    assert audit["plan_sha256"] == value.sha256


def test_source_has_no_hand_authored_sentence_frame_or_model_letter_synthesis():
    source = Path("experiments/plan_conditioned_exact_search_20260913.py").read_text()
    assert "compile_slots" not in source
    assert "left_text" not in source
    assert "right_text" not in source


def test_plan_conditioned_scorer_counts_the_new_child_unit_once():
    from experiments.plan_conditioned_exact_search_20260913 import PlanConditionedScorer

    value = plan()

    class Base:
        def word_delta(self, *args):
            return 0.0

    scorer = PlanConditionedScorer(value, Base(), fact_weight=10.0)
    assert scorer.word_delta(("baker", "checks", "oven"), (), "L", "oven", "append") == 10.0
    assert scorer.word_delta(("baker",), ("serves", "bread"), "R", "bread", "prepend") == 10.0
