import pytest

from experiments import cleaning_causal_order_voice_channels_20260913 as search
from experiments import cleaning_causal_voice_validator_20260913 as validator


@pytest.mark.parametrize("text,order,voice", [
    ("deli owners wash trays because they are soiled", "reason_last", "active"),
    ("deli ovens are cleaned by workers because they are soiled", "reason_last", "passive"),
    ("because the oven is soiled owners clean it", "condition_first", "active"),
    ("because the ovens are soiled owners wash them", "condition_first", "active"),
    ("because the oven is soiled it is cleaned by workers", "condition_first", "passive"),
])
def test_order_and_voice_have_independent_semantic_parse(text, order, voice):
    parses = validator.parse_complete(text.split())
    assert parses
    parsed = parses[0]
    assert parsed["order"] == order and parsed["voice"] == voice
    assert parsed["semantic_relation_valid"]
    assert parsed["causal_witness"]["same_patient_identity"]


def test_passive_pronoun_resolves_to_patient_despite_intervening_plural_agents():
    [parsed] = validator.parse_complete("the ovens are washed by workers because they are dirty".split())
    assert parsed["observation"]["patient"]["head"] == "ovens"
    assert parsed["observation"]["patient"]["syntactic_role"] == "patient"
    assert parsed["action"]["agent"]["head"] == "workers"
    assert parsed["observation"]["patient"]["id"] != parsed["action"]["agent"]["id"]
    assert validator.parse_complete("because the ovens are soiled owners wash it".split()) == []
    assert validator.parse_complete("because the ovens are soiled owners wash they".split()) == []
    assert validator.parse_complete("the oven are washed by workers because it is dirty".split()) == []


def test_causal_counterfactuals_fail_without_changing_syntax():
    [wrong_action] = validator.parse_complete("deli owners soil trays because they are dirty".split())
    assert wrong_action["complete"] and not wrong_action["semantic_relation_valid"]
    [no_condition] = validator.parse_complete("deli owners wash trays because they are clean".split())
    assert no_condition["complete"] and not no_condition["semantic_relation_valid"]


def test_full_constituent_discovery_keeps_condition_first_ineligible():
    channels, audit = search.discover_channels()
    assert channels
    assert {row["mode"] for row in channels} == {"reason_last_active", "reason_last_passive"}
    assert all(row["indexed_pairs"] >= 5 for row in channels)
    for row in audit:
        if row["mode"].startswith("condition_first"):
            assert row["selected_channels"] == 0
            assert set(row["indexed_depth_distribution"]) == {0}
            assert row["condition_first_endpoint_certificate"]


def test_each_selected_layout_has_independent_causal_parse_and_real_depth():
    channels, _ = search.discover_channels()
    # Cover every lexical agent/patient binding and each complete layout.
    for row in channels:
        for slots in search.surfaces(row).values():
            words = tuple(slot[0] for slot in slots)
            parses = validator.parse_complete(words)
            assert parses and all(parsed["semantic_relation_valid"] for parsed in parses), words
    for mode in ("reason_last_active", "reason_last_passive"):
        row = next(channel for channel in channels if channel["mode"] == mode)
        result = search.chart(next(iter(search.surfaces(row).values())))
        assert result["deepest"]["depth"] >= 5
        assert result["boundary_depth_distribution"]["left:4"] > 0


def test_unknown_generator_only_human_cannot_self_certify():
    assert validator.parse_complete("deli owlers wash trays because they are soiled".split()) == []


def test_free_center_and_island_prune_remain_live():
    good = search.chart((("ab",), ("cd",), ("cba",)))
    assert good["records"][0]["tape"] == "abcdcba"
    bad = search.chart((("ab",), ("cd",), ("dc",), ("ba",)))
    assert not bad["records"]
    assert bad["stats"]["proper_island_prunes"] == 1
