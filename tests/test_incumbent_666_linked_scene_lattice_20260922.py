import json

from experiments.incumbent_666_linked_scene_lattice_20260922 import OUT


def test_typed_writes_layer_is_bounded_and_records_online_reverse_obstruction():
    payload = json.loads(OUT.read_text())
    row = payload["rows"][0]
    attempt = row["typed_production_attempt"]
    assert attempt["normalized_windows"] == {"left": [140, 212], "right": [454, 526]}
    assert attempt["raw_windows"] == {"left": [187, 285], "right": [621, 718]}
    assert attempt["predicate_family"]["predicate"] == "writes"
    assert attempt["production_count"] == 4
    assert attempt["production_count"] <= attempt["max_productions"] == 8
    assert all(item["letters_per_side"] == 72 for item in attempt["attempts"])
    assert all(item["complete_scene"] is True for item in attempt["attempts"])
    first = attempt["attempts"][0]
    assert first["cursor"] == 0
    assert first["residual"]
    assert first["owner"] == "reverse-residual"
    assert first["trace"][0]["leading_shell_space"] == " "
    assert first["trace"][0]["trailing_shell_space"] == " "
    assert first["trace"][0]["active_entity_before"] == "Noel"
    assert first["grammar_state"]["active_entity"] == "Ari"
    assert first["grammar_state"]["reused_frames"] == []
    assert first["grammar_state"]["reused_clauses"] == []
    assert first["trace"][0]["clause_owner"] == "nora writes ari."
    assert first["trace"][0]["used_frames"] == ["nora|writes"]


def test_spacing_shell_is_preserved_in_candidate_assembly_and_alternate_seam_is_named():
    payload = json.loads(OUT.read_text())
    row = payload["rows"][0]
    attempt = row["typed_production_attempt"]
    spacing = attempt["primary_spacing"]
    assert spacing["leading_shell_spaces_preserved"] is True
    assert spacing["trailing_shell_spaces_preserved"] is True
    assert spacing["spacing_shell_preserved"] is True
    assert attempt["primary_candidate_sha256"]
    assert attempt["admission"]["accepted"] is False
    assert attempt["admission"]["exact_character_closure"] is False
    assert attempt["primary_candidate_sha256"] == "6a4a7b89b2d546d95bb61a6d88b67dfbd5a06e9e7b06d0a95a402bbbc0b943fb"
    alternate = attempt["alternate_seam_attempt"]
    assert alternate["normalized_windows"] == {"left": [127, 140], "right": [526, 539]}
    assert alternate["raw_windows"] == {"left": [171, 188], "right": [719, 736]}
    assert alternate["cursor"] == 3
    assert alternate["residual"]
    assert row["promotion_status"]["promoted"] is False
    assert row["next_operator"].startswith("switch to actual alternate seam")
