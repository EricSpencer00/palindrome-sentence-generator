import json
from pathlib import Path

from experiments.semantic_plan_character_trie_product_20260922 import (
    independent_validate,
    run,
    semantic_plan_pairs,
)


ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "runs" / "semantic-plan-character-trie-product-20260922.json"


def test_plans_predeclare_complete_agreed_valency_and_distinct_parses():
    pairs = semantic_plan_pairs()
    assert pairs
    for left, right in pairs:
        assert left.subject_number == left.finite_number
        assert right.subject_number == right.finite_number
        assert "required" in left.valency
        assert "required" in right.valency
        assert left.event != right.event
        assert left.parse_signature != right.parse_signature
        assert any(slot.role == "subject" for slot in left.slots)
        assert any(slot.role == "subject" for slot in right.slots)


def test_bounded_product_is_online_and_persists_cursor_frontier():
    result = run(max_states=1_200, max_survivors=2)
    assert result["construction"]["semantic_plan_selected_before_lexical_emission"]
    assert result["construction"]["complete_phrase_bank_built"] is False
    assert result["construction"]["word_boundaries_may_stagger"]
    assert result["construction"]["morphology_stack_or_r_equals_s"] is False
    assert result["stats"]["states_visited"] > 0
    assert result["deepest_semantic_character_frontier"]
    if not result["survivors"]:
        assert result["next_representation"]["lexicon_policy"].startswith("freeze")
        assert "dependency-chart" in result["next_representation"]["name"]


def test_remote_artifact_records_independent_validation_contract():
    payload = json.loads(ARTIFACT.read_text())
    assert payload["provenance"]["host"] == "hst-bench"
    assert payload["acceptance_gate"]["independent_validation_every_survivor"]
    assert payload["stats"]["state_budget_total"] == 250_000
    assert payload["deepest_semantic_character_frontier"]
    for row in payload["survivors"]:
        check = row["independent_validation"]
        assert row["all_gates_pass"]
        assert check["letters"] > 38
        assert check["two_pointer_exact"] and check["hashes_agree"]
        assert check["left_parse"]["ok"] and check["right_parse"]["ok"]


def test_independent_validator_rejects_nonmatching_surface():
    left, right = semantic_plan_pairs()[0]
    left_words = [slot.choices[0].surface for slot in left.slots]
    right_words = [slot.choices[0].surface for slot in right.slots]
    rendered = " ".join(left_words).capitalize() + "; " + " ".join(right_words) + "."
    audit = independent_validate(rendered, left, right)
    assert audit["left_parse"]["ok"] and audit["right_parse"]["ok"]
    assert not audit["two_pointer_exact"]
