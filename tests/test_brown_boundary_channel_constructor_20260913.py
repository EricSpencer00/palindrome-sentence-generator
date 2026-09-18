from experiments.brown_boundary_channel_constructor_20260913 import emit_pair, overlap_ok, render, search
from llm_palindrome.admission import normalize_letters
from pathlib import Path


def test_emission_checks_shifted_word_boundaries():
    assert overlap_ok("a red", "der a")[0]
    assert not overlap_ok("a red", "blue sky")[0]
    assert emit_pair("a red", "der a")["accepted"]


def test_search_is_explicitly_diagnostic_only_and_preserves_rejections():
    result = search(beam=12, limit_per_role=18)
    assert result["status"] == "diagnostic_only_missing_feature_grammar_witness"
    assert result["method"] == "brown_compiled_boundary_channel"
    assert result["role_skeleton"]["left"] == result["role_skeleton"]["right"]
    assert result["stats"]["partial_rejections"] > 0
    assert result["candidate_use"].startswith("forbidden")
    assert result["admitted"] == []
    for row in result["closures"]:
        assert row["independent_exact"] == (normalize_letters(row["rendered"]) == normalize_letters(row["rendered"])[::-1])
        assert len(row["provenance"]) == len(result["role_skeleton"]["left"])


def test_render_is_not_word_order_mirror_helper():
    text = render(("a", "red"), ("a", "der"))
    assert text.endswith(".")
    assert normalize_letters(text) == normalize_letters(text)[::-1]


def test_prior_pos_only_artifact_is_marked_ineligible_for_readers():
    marker = Path("runs/brown-boundary-channel-constructor-2026-09-13/INVALIDATED-NOT-A-CONSTRUCTOR.md")
    assert marker.is_file()
    assert "must not be shown to readers" in marker.read_text()
