from itertools import product

import pytest

from experiments import typed_causal_constituent_channels_20260913 as constructor
from experiments import causal_temperature_validator_20260913 as validator


def test_causal_validation_predicts_effect_instead_of_accepting_a_frame_name():
    positive = validator.causal_state_witness("cool", "hot")
    assert positive["valid"]
    assert positive["predicted_deviation"] < positive["initial_deviation"]
    assert not validator.causal_state_witness("heat", "hot")["valid"]
    assert not validator.causal_state_witness("cool", "cold")["valid"]
    assert validator.causal_state_witness("warm", "cold")["valid"]
    assert not validator.causal_state_witness("cool", "hot", referent_bound=False)["valid"]
    assert not validator.causal_state_witness("cool", "hot", target_type="person")["valid"]


def test_full_reparse_validates_causal_relation_and_agreement():
    words = "doctors cool bowls because their contents include hot cod".split()
    [parsed] = validator.parse_complete(words)
    assert parsed["semantic_relation_valid"]
    assert parsed["reason"]["possessor"] == "bowls"
    [wrong_effect] = validator.parse_complete("doctors heat bowls because their contents include hot cod".split())
    assert wrong_effect["complete"] and not wrong_effect["semantic_relation_valid"]
    assert validator.parse_complete("doctor cool bowls because their contents include hot cod".split()) == []
    assert validator.parse_complete("doctors cool bowls because their contents include hot engines".split()) == []
    assert validator.parse_complete("doctors cool bowls because their reports include hot cod".split()) == []


def test_complete_constituent_index_discovers_and_replays_five_pairs():
    channels = constructor.discover_channels()
    assert channels
    assert all(row["indexed_pairs"] >= 5 for row in channels)
    assert all(row["causal_effect"]["corrective"] for row in channels)
    with pytest.raises(ValueError):
        constructor.discover_channels(4)
    for row in channels:
        for slots in constructor.layouts(row).values():
            result = constructor.chart(slots)
            assert result["deepest"]["depth"] >= 5
            assert result["state_depth_distribution"][5] > 0


def test_every_installed_layout_has_independent_complete_causal_parse():
    for row in constructor.discover_channels():
        for slots in constructor.layouts(row).values():
            words = tuple(slot[0] for slot in slots)
            parses = validator.parse_complete(words)
            assert parses and all(parse["semantic_relation_valid"] for parse in parses)


def test_validator_does_not_accept_generator_only_actor(monkeypatch):
    monkeypatch.setattr(constructor, "ACTORS", ("docto",))
    assert constructor.discover_channels()
    assert validator.parse_complete("docto cool bowls because their contents include hot cod".split()) == []


def test_free_midpoint_and_online_multiword_island_guard():
    free = constructor.chart((("ab",), ("cd",), ("cba",)))
    assert free["records"][0]["tape"] == "abcdcba"
    assert free["records"][0]["matched_depth"] == 3
    assert free["records"][0]["center_characters"] == 1
    blocked = constructor.chart((("ab",), ("cd",), ("dc",), ("ba",)))
    assert blocked["records"] == []
    assert blocked["stats"]["proper_island_prunes"] == 1


def test_full_finite_run_is_scoped_and_does_not_promote_unreviewed_text():
    report = constructor.run()
    assert report["states_exhausted"]
    assert report["actual_eligible_products"] == report["products"] > 0
    assert any(row["length_range"][1] >= 100 for row in report["rows"])
    assert report["promoted_candidates"] == []
    assert not report["config"]["fixed_clause_boundary_center"]
    assert not report["config"]["source_tape_reflection"]
