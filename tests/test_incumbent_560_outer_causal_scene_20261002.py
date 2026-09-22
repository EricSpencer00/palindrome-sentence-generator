from experiments.incumbent_560_outer_causal_scene_20261002 import build_payload


def test_outer_causal_scene_is_exact_and_longer() -> None:
    payload = build_payload()
    row = payload["rows"][0]

    assert row["audit"]["letters"] == 568
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["byte_pointer_exact"]
    assert row["audit"]["project_validator_exact"]
    assert row["growth_over_parent"] == 8
    assert row["live_state"]["final_residual"] == ""


def test_outer_scene_does_not_hide_inherited_shortcut_failure() -> None:
    payload = build_payload()
    strict = payload["rows"][0]["strict_admission"]

    assert strict["proper_span_count"] == 65
    assert not strict["no_self_palindromic_proper_multiword_span"]
    assert not strict["no_repeated_nontrivial_unit"]
    assert not strict["distinct_content_words"]
    assert not strict["shortcut_clean"]
    assert payload["stats"]["shortcut_clean_children"] == 0
