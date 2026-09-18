from experiments.guide_cross_boundary_material import (
    ScreenItem,
    candidate_items,
    parse_reply,
    partial_result,
)


def test_parse_reply_accepts_a_word_preserving_selection():
    item = ScreenItem("x", "candidate", None, "step on no pets", "no pets step on")
    parsed, error = parse_reply(
        '{"choice":"a","display":"Step on, no pets.",'
        '"interpretation":"A warning not to step on pets.","reason":"clear command"}', item)
    assert error is None
    assert parsed["choice"] == "a"


def test_parse_reply_rejects_resegmentation_and_bad_neither():
    item = ScreenItem("x", "candidate", None, "step on no pets", "no pets step on")
    parsed, error = parse_reply(
        '{"choice":"a","display":"Step onno, pets.",'
        '"interpretation":"x","reason":"x"}', item)
    assert parsed is None
    assert error == "display_changed_words_or_word_order"
    parsed, error = parse_reply(
        '{"choice":"neither","display":"x","interpretation":null,"reason":"x"}', item)
    assert parsed is None
    assert error == "neither_must_not_supply_display_or_interpretation"


def test_candidate_items_hide_arm_labels_from_prompt_inputs():
    materials = {"variants": [
        {"source_arm": "cross_boundary_only", "block_id": "CR01", "condition": "a", "plain": "a"},
        {"source_arm": "cross_boundary_only", "block_id": "CR01", "condition": "b", "plain": "b"},
    ]}
    items = candidate_items(materials)
    assert len(items) == 1
    assert items[0].a == "a"
    assert items[0].b == "b"
    assert items[0].source_arm == "cross_boundary_only"


def test_partial_result_records_the_frozen_run_identity(tmp_path):
    materials = tmp_path / "materials.json"
    materials.write_text("{}")
    partial = partial_result(materials, "model", 7, {"digest": "x"}, [])
    assert partial["status"] == "incomplete_development_only_screen"
    assert partial["model_requested"] == "model"
    assert partial["seed"] == 7
    assert partial["records"] == []
