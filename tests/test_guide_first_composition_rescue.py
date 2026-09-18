import pytest

from experiments.guide_first_composition_rescue import blocks, parse_reply


def test_parse_reply_accepts_a_grounded_choice_and_neither():
    assert parse_reply('{"choice":"a","intended_interpretation":"A traveller waits.","reason":"It has a subject."}') == {
        "choice": "a", "intended_interpretation": "A traveller waits.", "reason": "It has a subject.",
    }
    assert parse_reply('```json\n{"choice":"neither","intended_interpretation":null,"reason":"No reading."}\n```')["choice"] == "neither"


def test_parse_reply_rejects_uninterpreted_or_malformed_choice():
    with pytest.raises(ValueError):
        parse_reply('{"choice":"a","intended_interpretation":null,"reason":"x"}')
    with pytest.raises(ValueError):
        parse_reply('{"choice":"neither","intended_interpretation":"Actually yes","reason":"x"}')


def test_blocks_requires_both_counter_orders():
    materials = {"variants": [
        {"id": "C1", "block_id": "B01", "condition": "order_a_outer", "plain": "a"},
        {"id": "C2", "block_id": "B01", "condition": "order_b_outer", "plain": "b"},
        {"id": "C3", "block_id": "B01", "condition": "constituent", "plain": "c"},
    ]}
    assert blocks(materials)[0]["a"]["id"] == "C1"
    with pytest.raises(ValueError):
        blocks({"variants": materials["variants"][:1]})
