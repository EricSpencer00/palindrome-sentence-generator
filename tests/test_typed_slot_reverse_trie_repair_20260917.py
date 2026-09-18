from experiments.typed_slot_reverse_trie_repair_20260917 import run


def test_typed_repair_requires_the_selected_right_phrase_to_match_live_seam():
    artifact = run()
    assert artifact["stats"]["typed_variants"] == 12
    assert artifact["stats"]["exact"] == 0
    assert artifact["stats"]["rendered_near_misses"] == len(artifact["candidates"])
    assert all(not row["audit"]["exact"] for row in artifact["candidates"])

