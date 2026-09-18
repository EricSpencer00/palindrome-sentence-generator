import inspect
from pathlib import Path

import pytest

from experiments.measurement_cleft_source_parser_20260913 import parse_source
from experiments.measurement_cleft_final_parser_20260913 import parse_final, render_final
from experiments.measurement_cleft_residual_feasibility_20260913 import (
    productions, review, frontier_witnesses, discover, run,
)


@pytest.fixture(scope="module")
def report():
    return run()


def test_true_measurement_clefts_and_headed_equatives_end_in_finite_repair_predicate():
    paths = list(productions())
    assert {p["source_relation"]["mode"] for p in paths} == {"scalar_what_cleft", "headed_relative_equative"}
    assert {p["source_relation"]["property"] for p in paths} == {"height", "roughness", "firmness", "permeability"}
    for p in paths:
        assert p["target_tokens"][-1] == p["source_relation"]["finite_repair_predicate"]
        assert "that" in p["target_tokens"]
        if p["source_relation"]["mode"] == "headed_relative_equative":
            assert "whose" in p["target_tokens"]
        else:
            assert "what" in p["target_tokens"]
    code = inspect.getsource(productions)
    assert "[::-1]" not in code and "reversed(" not in code


def test_source_final_parsers_are_independent_and_tokenizations_are_not_shared():
    for name in ("measurement_cleft_source_parser_20260913.py", "measurement_cleft_final_parser_20260913.py"):
        code = Path("experiments", name).read_text()
        assert "from experiments" not in code and "productions(" not in code
        assert "source_relation" not in code
    p = next(productions())
    assert parse_source(p["source_tokens"]) and parse_final(p["target_tokens"])
    assert not parse_source(p["target_tokens"]) and not parse_final(p["source_tokens"])


def test_scalar_property_patient_equative_and_repair_gap_are_all_bound(report):
    for row in report["screen"]["reviews"]:
        source, = row["source_parse"]
        final, = row["independent_final_parse"]
        assert source["semantic_relation_valid"] and final["semantic_relation_valid"]
        assert source["patient_id"] == source["measure"]["patient_id"] == source["equative"]["patient_id"] == source["repair_relative"]["patient_id"]
        patient = final["patient"]["id"]
        assert patient == final["initial_relative"]["patient_id"]
        assert patient == final["equative_binding"]["left_patient_id"] == final["equative_binding"]["right_patient_id"]
        assert patient == final["finite_repair_relative"]["gap_patient_id"]
        measure = final["measured_comparison"]
        assert patient == measure["before_patient_id"] == measure["after_patient_id"]
        assert measure["property"] == final["finite_repair_relative"]["affected_property"]
        assert measure["before"] == source["measure"]["before"] and measure["after"] == source["measure"]["after"]
        assert measure["satisfied"]


def test_opposite_repair_effect_parses_but_fails_ordered_change():
    p = next(p for p in productions() if p["source_relation"]["property"] == "height"
             and p["source_relation"]["finite_repair_predicate"] == "reduces")
    for key, parser in (("source_tokens", parse_source), ("target_tokens", parse_final)):
        parsed, = parser(p[key][:-1] + ("raises",))
        assert not parsed["semantic_relation_valid"]


def test_equative_identity_and_repair_property_cannot_be_swapped():
    p = next(p for p in productions() if p["source_relation"]["property"] == "height"
             and p["source_relation"]["mode"] == "headed_relative_equative")
    for key, parser in (("source_tokens", parse_source), ("target_tokens", parse_final)):
        assert not parser(tuple("areas" if w == "ridges" else w for w in p[key]))
        assert not parser(p[key][:-1] + ("secures",))
        assert not parser(tuple("roughness" if w == "height" else w for w in p[key]))


def test_comparative_direction_copula_and_process_subject_must_match():
    p = next(p for p in productions() if p["source_relation"]["mode"] == "headed_relative_equative")
    for key, parser in (("source_tokens", parse_source), ("target_tokens", parse_final)):
        assert not parser(tuple("higher" if w == "lower" else w for w in p[key]))
        assert not parser(tuple("is" if w == "are" else w for w in p[key]))
        assert not parser(tuple("fresh" if w == "coarse" else w for w in p[key]))
        assert not parser(p[key] + ("again",))


def test_rendering_is_letter_preserving_in_both_cleft_forms():
    for p in productions():
        rendered = render_final(p["target_tokens"])
        assert "".join(c for c in rendered.lower() if c.isalpha()) == "".join(p["target_tokens"])
        if p["target_tokens"][:3] == ("none", "the", "less"):
            assert rendered.startswith("None the less, ")


def test_finite_accounting_matches_literal_full_source_target_paths(report):
    paths = list(productions())
    result = report["screen"]
    assert len(paths) == result["source_target_derivation_pairs"] == 84
    assert len({p["source_tokens"] for p in paths}) == result["distinct_source_analyses"] == 84
    assert len({p["target_tokens"] for p in paths}) == result["distinct_target_surfaces"] == 84
    assert len(result["reviews"]) == 84 and result["finite_exhausted"]
    assert result["actual_pair_depth_distribution"] == {0: 63, 1: 9, 2: 3, 3: 9}
    assert sum(m["derivation_pairs"] for m in result["outer_mismatches"]) == 84
    assert result["failure_counts"] == {"exact_letter_palindrome": 84, "length_band": 7}
    assert all(r["central_admission"]["lexicon_words"] for r in result["reviews"])


def test_deepest_actual_endpoint_mismatches_and_no_raw_boundary_qualification(report):
    result = report["screen"]
    deepest = [r for r in result["reviews"] if r["fringe_trace"]["actual_pairs"] == 3]
    assert len(deepest) == 9
    assert sum(r["fringe_trace"]["left"] == "t" and r["fringe_trace"]["right"] == "u" for r in deepest) == 6
    assert sum(r["fringe_trace"]["left"] == "b" and r["fringe_trace"]["right"] == "e" for r in deepest) == 3
    assert all(r["fringe_trace"]["mismatch_pair"] == 4 for r in deepest)
    assert result["reached_minimum_pair_derivations"] == result["live_two_sided_boundary_derivations_at_or_beyond_minimum"] == 0
    assert result["qualified_channels"] == 0 and result["channels"] == []


def test_frontiers_beyond_eleven_are_scanned_and_late_cut_is_not_prematurely_claimed():
    # Non-English mechanism fixture: one source cut appears only after depth12.
    left = "qwertyuiopazbcde"
    right = "u" + left[::-1]
    source = (left[:12], left[12:], "uv", right[:-6], right[-6:])
    target = (left, "uv", right)
    row = review({"source_tokens": source, "target_tokens": target, "source_relation": {"fixture": True}})
    witnesses = frontier_witnesses(row)
    assert len(left) == 16 and len(row["normalized"]) == 35
    assert row["fringe_trace"]["actual_pairs"] == 17
    assert [w["depth"] for w in witnesses] == [11, 12, 13, 14, 15, 16]
    assert witnesses[-1]["next_left"] == witnesses[-1]["next_right"] == "u"
    assert witnesses[0]["boundary_witness"]["crossed_left_disagreements"] == []
    assert witnesses[1]["boundary_witness"]["crossed_left_disagreements"] == []
    assert witnesses[2]["boundary_witness"]["crossed_left_disagreements"] == [12]
    assert witnesses[2]["boundary_witness"]["crossed_right_disagreements"] == [6]
    assert not any(w["two_live_boundary_sides"] for w in witnesses)  # English/semantic admission still fails.
    assert not row["eligible_for_external_provenance"]


def test_lone_odd_center_is_never_counted_as_a_further_outer_pair():
    words = ("qwertyuiopz", "v", "zpoiuytrewq")
    row = review({"source_tokens": words, "target_tokens": words, "source_relation": {"fixture": True}})
    assert row["fringe_trace"]["actual_pairs"] == 11
    assert row["fringe_trace"]["center_kind"] == "odd"
    assert frontier_witnesses(row) == []


def test_derivation_count_does_not_become_unique_surface_count():
    p = next(productions())
    result = discover([p, p])
    assert result["source_target_derivation_pairs"] == 2
    assert result["distinct_source_analyses"] == result["distinct_target_surfaces"] == 1


@pytest.mark.parametrize("kwargs", [{"minimum_pairs": 10}, {"minimum_letters": 5}])
def test_minimum_eleven_pairs_and_six_distinct_letters_cannot_be_relaxed(kwargs):
    with pytest.raises(ValueError):
        discover([], **kwargs)


@pytest.mark.parametrize("middle,parity", [("uv", "odd"), ("uvv", "even")])
def test_all_exact_fixture_closures_rendered_and_preserved_for_failed_review(middle, parity):
    words = ("qwertyb", middle, "ubytrewq")
    result = discover([{"source_tokens": words, "target_tokens": words, "source_relation": {"fixture": True}}])
    row, = result["closure_reviews"]
    assert row["fringe_trace"]["center_kind"] == parity
    assert row["rendered_diagnostic"] == " ".join(words).capitalize() + "."
    assert row["source_parse"] == row["independent_final_parse"] == []
    assert "independent_scalar_patient_repair_binding" in row["failures"]
    assert row["external_provenance"] == "not_checked"
    assert not row["eligible_for_external_provenance"] and not row["promoted"]


@pytest.mark.parametrize("words,check", [
    (("abc", "dog", "god", "cba"), "no_self_palindromic_proper_multiword_span"),
    (("abc", "level", "cba"), "no_self_palindromic_word"),
    (("dogs", "dogs"), "distinct_words"),
])
def test_central_shortcut_rejections_preserved(words, check):
    row = review({"source_tokens": words, "target_tokens": words, "source_relation": {"fixture": True}})
    assert not row["central_admission"][check]
    assert not row["promoted"]


def test_unqualified_screen_builds_no_product_interface_or_originality_claim(report):
    result = report["screen"]
    assert result["exact_closure_derivations"] == result["distinct_exact_closure_surfaces"] == 0
    assert not result["full_palindrome_product_compiled"] and not result["model_interface_implemented"]
    assert result["model_queries_executed"] == 0
    assert report["provenance"]["external_candidate_provenance_searches"] == 0
    assert report["provenance"]["external_review_required_before_promotion"]
    assert "non-cleft temporal comparison" in report["next_construction"]
