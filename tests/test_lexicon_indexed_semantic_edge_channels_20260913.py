from itertools import product

import pytest

from experiments import lexicon_indexed_semantic_edge_channels_20260913 as search
from experiments import semantic_edge_validator_20260913 as validator


def test_broad_inventory_join_is_typed_and_matches_at_least_three_letters():
    assert len(search.SUBJECTS) > 100
    assert len(search.NOUNS) > 200
    assert len(search.VERBS) > 25
    channels = search.discover_channels()
    assert channels
    assert all(depth >= 3 and noun.kind in search.VERBS[verb] for subject, verb, noun, depth in channels)
    assert all(subject[:3] == noun.word[::-1][:3] for subject, verb, noun, depth in channels)
    with pytest.raises(ValueError):
        search.discover_channels(min_pairs=2)


def test_discovered_three_pair_channel_replays_actual_characters():
    channels = search.discover_channels()
    subject, verb, noun, _ = next(row for row in channels if row[0] == "doctors" and row[1] == "inspect" and row[2].word == "cod")
    [(_, np_slots), *_] = list(search.noun_phrase_forms(noun))
    result = search.chart(search.complete_layouts(subject, verb, np_slots)["simple"])
    assert result["deepest"]["depth"] >= 3
    assert all(result["state_depth_distribution"][n] > 0 for n in (1, 2, 3))
    assert result["boundary_depth_distribution"]["right:3"] > 0


def test_shifted_boundary_positive_toy_and_island_prune():
    positive = search.chart((("ab",), ("cd",), ("cba",)))
    assert positive["records"][0]["tape"] == "abcdcba"
    assert positive["deepest"]["left_boundary_depths"] == (2,)
    assert positive["deepest"]["right_boundary_depths"] == (3,)
    negative = search.chart((("ab",), ("cd",), ("dc",), ("ba",)))
    assert negative["records"] == []
    assert negative["stats"]["proper_island_prunes"] == 1
    assert negative["prune_witnesses"][0]["matched_depth"] == 2


def test_reparser_has_independent_lexical_declarations():
    invented = search.Noun("cod", "food", "mass")
    channels = search.discover_channels(subjects=("docx",), nouns=(invented,), verbs={"inspect": {"food"}})
    assert channels  # The generator's local declaration alone permits this.
    assert validator.parse_surface(("docx", "inspect", "cod")) == []
    assert validator.parse_surface(("doctors", "inspect", "cod"))
    assert validator.parse_surface(("doctor", "inspect", "cod")) == []
    assert validator.parse_surface(("doctors", "inspects", "cod")) == []
    assert validator.parse_surface(("doctors", "read", "cod")) == []


def test_independent_parser_recognizes_relative_source_and_location():
    text = "doctors who reviewed detailed reports from careful authors in quiet offices slowly inspect fresh cod"
    parses = validator.parse_surface(text.split())
    assert parses and parses[0]["relative_subject_shared"]
    assert parses[0]["relative"]["location"]["kind"] == "place"
    assert not validator.parse_surface(text.replace("offices", "cod").split())


def test_every_discovered_channel_has_a_complete_independent_parse():
    for subject, verb, noun, _ in search.discover_channels():
        for _, np_slots in search.noun_phrase_forms(noun):
            for slots in search.complete_layouts(subject, verb, np_slots).values():
                words = tuple(slot[0] for slot in slots)
                assert validator.parse_surface(words), words


def test_article_forms_agree_with_every_adjective_realization():
    noun = search.Noun("engine", "artifact", "singular")
    for _, slots in search.noun_phrase_forms(noun):
        for words in product(*slots):
            assert validator.parse_surface(("workers", "inspect", *words))
