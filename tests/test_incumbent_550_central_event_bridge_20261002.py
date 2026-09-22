from experiments.incumbent_550_central_event_bridge_20261002 import (
    EXPECTED_SHA256,
    build_payload,
)


def test_central_bridge_is_exact_and_beats_rough_control() -> None:
    payload = build_payload()
    row = payload["rows"][0]

    assert row["audit"]["letters"] == 560
    assert row["audit"]["sha256_forward"] == EXPECTED_SHA256
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["byte_pointer_exact"]
    assert row["audit"]["project_validator_exact"]
    assert payload["stats"]["children_over_556_control"] == 1


def test_central_bridge_has_distinct_events_and_staggered_boundaries() -> None:
    row = build_payload()["rows"][0]
    structure = row["structural_audit"]

    assert row["replacement"]["new_lexical_content"] == ["Ari", "Ira"]
    assert structure["names_distinct"]
    assert not structure["self_palindromic_word_used"]
    assert not structure["catalogue_text_used"]
    assert structure["proper_palindromic_multiword_span_used"]
    assert not structure["shortcut_clean"]
    assert structure["working_track_only"]
    assert not structure["word_boundaries_reflect_one_to_one"]
    assert row["live_state"]["residual"] == "s"
    assert row["live_state"]["residual_after_shell"] == ""
