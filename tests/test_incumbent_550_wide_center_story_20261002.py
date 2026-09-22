from experiments.incumbent_550_wide_center_story_20261002 import build_payload


def test_wide_center_story_is_saved_and_independently_exact() -> None:
    payload = build_payload()
    row = payload["rows"][0]

    assert row["audit"]["letters"] == 532
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["byte_pointer_exact"]
    assert row["audit"]["project_validator_exact"]
    assert row["replacement"]["old_letters"] == 186
    assert row["replacement"]["new_letters"] == 168


def test_wide_center_story_is_not_promoted_past_shortcut_gate() -> None:
    payload = build_payload()
    admission = payload["rows"][0]["strict_admission"]

    assert admission["proper_span_count"] == 64
    assert not admission["no_self_palindromic_proper_multiword_span"]
    assert not admission["no_repeated_nontrivial_unit"]
    assert not admission["distinct_content_words"]
    assert not admission["shortcut_clean"]
    assert payload["stats"]["shortcut_clean_children"] == 0
