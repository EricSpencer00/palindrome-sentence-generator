from pathlib import Path

import pytest

from experiments.argument_order_residual_feasibility_20260913 import review, run, screen, source_paths
from experiments.ownership_coordination_validator_20260913 import parse_sentence, render_sentence


@pytest.fixture(scope="module")
def result():
    return run()


def test_ownership_is_outside_final_gap_and_bound_to_same_repaired_artwork():
    path = next(source_paths())
    parsed, = parse_sentence(path["words"])
    assert parsed["semantic_relation_valid"]
    assert parsed["ownership"]["surface_span"] == path["source_analysis"]["possessor_span"]
    assert parsed["ownership"]["surface_span"][1] < path["words"].index("and")
    assert parsed["reference"]["compatible_antecedents"] == ["owned_collection"]
    assert parsed["second_event"]["patient"] == parsed["ownership"]["patient_id"]
    assert parsed["first_event"]["patient"] == parsed["second_event"]["patient"]
    assert parsed["semantic_witness"]["same_owned_patient"]


def test_whole_sentence_parser_is_independently_declared():
    code = Path("experiments/ownership_coordination_validator_20260913.py").read_text()
    assert "from experiments" not in code
    assert "source_paths" not in code
    assert "SOURCE_" not in code
    # Whole-grammar acceptance is not membership in the generating inventory.
    other = "conservators arrange foreign patrons torn prints and mend large rips in this high gloss blue artwork".split()
    assert parse_sentence(other)[0]["semantic_relation_valid"]


@pytest.mark.parametrize("old,new,expected_parse", [
    ("repair", "worsen", True),
    ("this", "their", True),
    ("artists", "paintings", False),
    ("paintings", "workers", False),
    ("and", "because", False),
])
def test_parser_rejects_wrong_event_reference_owner_patient_or_clause_structure(old, new, expected_parse):
    words = list(next(source_paths())["words"])
    words[words.index(old)] = new
    parsed = parse_sentence(words)
    assert bool(parsed) == expected_parse
    assert not any(p["semantic_relation_valid"] for p in parsed)


def test_complete_consumption_and_unsupported_extra_artwork_antecedent_rejected():
    words = next(source_paths())["words"]
    assert not parse_sentence(words + ("again",))
    cut = words.index("and")
    assert not parse_sentence(words[:cut] + ("and", "posters") + words[cut:])


def test_source_is_composed_lexically_without_reversing_source_tape():
    import inspect
    source = inspect.getsource(source_paths)
    assert "[::-1]" not in source and "reversed(" not in source
    assert '"and", "repair"' in source
    for path in source_paths():
        assert path["words"][-5:] == ("this", "high", "gloss", "red", "art")
        assert path["source_analysis"]["owner_outside_pre_art_gap"]


def test_production_not_fixture_has_nine_actual_pairs_and_432_continuations(result):
    report = result["argument_order_screen"]
    assert report["actual_pair_depth_distribution"] == {9: 432}
    channel, = report["channels"]
    assert channel["actual_probe_pairs"] == 8
    assert channel["continued_actual_pairs"] == 9
    assert channel["distinct_next_pair_compatible_surfaces"] == 432
    assert channel["six_surface_alternative_probe_passed"]
    assert channel["distinct_next_pair_letters"] == ["o"]
    assert not channel["six_distinct_next_letters_passed"]
    for witness in channel["witnesses"]:
        assert witness["actual_pairs"] == 9
        assert witness["next_left"] == witness["next_right"] == "o"
        assert witness["reference"]["resolved"] == "owned_collection"


def test_finite_accounting_equals_literal_independent_enumeration(result):
    report = result["argument_order_screen"]
    paths = list(source_paths())
    words = {tuple(p["words"]) for p in paths}
    surfaces = {render_sentence(w) for w in words}
    assert len(paths) == report["source_derivations"] == 432
    assert len(words) == len(surfaces) == report["distinct_rendered_surfaces"] == 432
    assert len(report["reviews"]) == 432 and report["states_exhausted"]
    for row in report["reviews"]:
        tape = row["normalized"]
        assert tape[:9] == tape[-9:][::-1]
        assert tape[9] == "r" and tape[-10] == "l"
        assert not row["central_admission"]["exact_letter_palindrome"]
        assert all(v for k, v in row["central_admission"].items() if k != "exact_letter_palindrome")
        assert any(p["semantic_relation_valid"] for p in row["independent_parses"])
    assert report["full_exact_completion_surfaces"] == report["central_and_semantic_survivors"] == 0


def test_compare_fixed_owner_control_and_fail_closed_transport(result):
    assert result["fixed_owner_control"]["actual_pair_depth_distribution"] == {7: 192}
    report = result["argument_order_screen"]
    assert report["outer_mismatches"] == [{"pair": 10, "left": "r", "right": "l", "source_derivations": 432}]
    assert report["qualified_eight_pair_six_surface_probe_channels"] == 1
    assert report["qualified_eight_pair_six_letter_probe_channels"] == 0
    assert not report["local_infill_transport_eligible"]
    assert not result["model_transport_implemented"]
    assert result["model_queries_executed"] == 0
    assert result["promoted_candidates"] == []


@pytest.mark.parametrize("kwargs", [{"probe_pairs": 7}, {"minimum_alternatives": 5}])
def test_minimum_eight_pair_six_alternative_probe_cannot_be_relaxed(kwargs):
    with pytest.raises(ValueError):
        screen([], **kwargs)


def test_five_surface_control_does_not_claim_six_alternatives():
    paths = list(source_paths())[:5]
    report = screen(paths)
    assert report["source_derivations"] == report["distinct_rendered_surfaces"] == 5
    assert report["qualified_eight_pair_six_surface_probe_channels"] == 0


def test_derivation_multiplicity_does_not_inflate_surface_alternatives():
    path = next(source_paths())
    report = screen([path] * 6)
    assert report["source_derivations"] == 6
    assert report["distinct_rendered_surfaces"] == 1
    assert report["qualified_eight_pair_six_surface_probe_channels"] == 0


@pytest.mark.parametrize("words,parity", [
    (("qwertyb", "uv", "ubytrewq"), "odd"),
    (("qwertyb", "uvv", "ubytrewq"), "even"),
])
def test_every_exact_fixture_closure_rendered_and_independently_rejected(words, parity):
    # Mechanism fixtures only: no fixture can stand in for production English.
    report = screen([{"words": words, "source_analysis": {"fixture": True}}])
    row, = report["closure_reviews"]
    assert row["central_admission"]["exact_letter_palindrome"]
    assert row["fringe_trace"]["center_kind"] == parity
    assert row["rendered_diagnostic"] == " ".join(words).capitalize() + "."
    assert row["independent_parses"] == []
    assert "independent_whole_sentence_semantics" in row["failures"]
    assert not row["eligible_for_external_provenance"] and not row["promoted"]
    assert row["external_provenance"] == "not_checked"


@pytest.mark.parametrize("words,check", [
    (("abc", "dog", "god", "cba"), "no_self_palindromic_proper_multiword_span"),
    (("abc", "level", "cba"), "no_self_palindromic_word"),
    (("dogs", "dogs"), "distinct_words"),
])
def test_central_shortcut_failures_remain_in_rendered_reviews(words, check):
    row = review({"words": words, "source_analysis": {"fixture": True}})
    assert not row["central_admission"][check]
    assert check in row["failures"]
    assert not row["promoted"]
