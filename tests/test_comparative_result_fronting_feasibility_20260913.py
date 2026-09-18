import inspect
from pathlib import Path

import pytest

from experiments.comparative_fronting_source_parser_20260913 import parse_source
from experiments.comparative_fronting_final_parser_20260913 import parse_final, render_final
from experiments.comparative_result_fronting_feasibility_20260913 import productions, review, discover, run


@pytest.fixture(scope="module")
def report():
    return run()


def test_initial_result_and_final_material_are_joint_typed_roles():
    paths = list(productions())
    assert {p["source_relation"]["repair"] for p in paths} == {"polishing", "sealing", "tightening", "sanding"}
    assert len({p["source_relation"]["comparison"] for p in paths}) == 5
    assert {p["source_relation"]["concessive"] for p in paths} == {True, False}
    for p in paths:
        words = p["target_tokens"]
        offset = 3 if p["source_relation"]["concessive"] else 0
        assert words[offset:words.index("than")] == p["source_relation"]["comparison"]
        assert words[-1] in {"work", "panels", "vessel"}
    code = inspect.getsource(productions)
    assert "[::-1]" not in code and "reversed(" not in code


def test_source_and_final_parsers_are_separate_declarations():
    for name in ("comparative_fronting_source_parser_20260913.py", "comparative_fronting_final_parser_20260913.py"):
        code = Path("experiments", name).read_text()
        assert "from experiments" not in code
        assert "productions(" not in code and "source_relation" not in code
    p = next(productions())
    assert parse_source(p["source_tokens"]) and parse_final(p["target_tokens"])
    assert not parse_source(p["target_tokens"]) and not parse_final(p["source_tokens"])


def test_all_comparatives_compare_two_states_of_the_same_repaired_patient(report):
    for row in report["screen"]["reviews"]:
        source, = row["source_parse"]
        final, = row["independent_final_parse"]
        assert source["semantic_relation_valid"] and final["semantic_relation_valid"]
        assert source["patient_id"] == source["repair"]["patient_id"] == source["comparison"]["patient_id"]
        assert final["patient"]["id"] == final["initial_property"]["patient_id"] == final["repair_process"]["controlled_patient_id"]
        measure = final["fronted_comparative"]
        assert measure["current_patient_id"] == measure["prior_patient_id"] == final["patient"]["id"]
        assert measure["current_state"] == final["predicted_state"] == source["after"]
        assert measure["prior_state"] == source["before"]
        assert final["causal_witness"] == {"same_patient": True, "same_property": True, "material_compatible": True, "ordered_change": True}


def test_reversed_repair_effect_fails_ordinary_comparative_semantics():
    p = next(p for p in productions() if p["source_relation"]["repair"] == "tightening")
    for key, parser in (("source_tokens", parse_source), ("target_tokens", parse_final)):
        words = tuple("loosening" if w == "tightening" else w for w in p[key])
        result, = parser(words)
        assert not result["semantic_relation_valid"]
        if key == "target_tokens":
            assert result["predicted_state"] == 0
            assert not result["fronted_comparative"]["satisfied"]


def test_equal_comparison_direction_on_different_property_still_fails():
    p = next(p for p in productions() if p["source_relation"]["repair"] == "polishing")
    for key, parser in (("source_tokens", parse_source), ("target_tokens", parse_final)):
        words = tuple("lower" if w == "smoother" else w for w in p[key])
        result, = parser(words)
        assert not result["semantic_relation_valid"]
        if key == "target_tokens":
            assert result["fronted_comparative"]["satisfied"]  # Numerical decrease alone is insufficient.
            assert not result["causal_witness"]["same_property"]


def test_material_and_number_counterfactuals_are_rejected():
    p = next(p for p in productions() if p["source_relation"]["repair"] == "tightening"
             and p["source_relation"]["material"] == "metal")
    a = tuple("glasswork" if w == "metalwork" else w for w in p["source_tokens"])
    b = tuple("glass" if w == "metal" else w for w in p["target_tokens"])
    assert not parse_source(a)[0]["semantic_relation_valid"]
    assert not parse_final(b)[0]["semantic_relation_valid"]
    assert not parse_source(tuple("is" if w == "are" else w for w in p["source_tokens"]))
    assert not parse_final(tuple("is" if w == "are" else w for w in p["target_tokens"]))


def test_baseline_agent_and_extra_sentence_cannot_be_inserted():
    p = next(productions())
    for key, parser in (("source_tokens", parse_source), ("target_tokens", parse_final)):
        words = p[key]
        assert not parser(tuple("after" if w == "before" else w for w in words))
        assert not parser(words + ("again",))
        place = words.index("is")
        assert not parser(words[:place] + ("by", "artists") + words[place:])


def test_rendering_preserves_tape_and_marks_fronted_comparison_process():
    p = next(productions())
    rendered = render_final(p["target_tokens"])
    assert "than before, after" in rendered and "hours, is the" in rendered
    assert "".join(c for c in rendered.lower() if c.isalpha()) == "".join(p["target_tokens"])
    concessive = next(p for p in productions() if p["source_relation"]["concessive"])
    assert render_final(concessive["target_tokens"]).startswith("None the less, ")


def test_finite_accounting_matches_literal_surface_enumeration(report):
    paths = list(productions())
    result = report["screen"]
    assert len(paths) == result["source_target_derivation_pairs"] == 46
    assert len({p["target_tokens"] for p in paths}) == result["distinct_target_surfaces"] == 46
    assert len({p["source_tokens"] for p in paths}) == result["distinct_source_analyses"] == 46
    assert len(result["reviews"]) == 46 and result["finite_exhausted"]
    assert result["actual_pair_depth_distribution"] == {0: 43, 1: 2, 4: 1}
    assert sum(m["derivation_pairs"] for m in result["outer_mismatches"]) == 46
    assert result["failure_counts"] == {"distinct_words": 5, "exact_letter_palindrome": 46}


def test_deepest_endpoint_is_exactly_less_permeable_vessel_and_fails_at_five(report):
    row, = [r for r in report["screen"]["reviews"] if r["fringe_trace"]["actual_pairs"] == 4]
    assert row["target_tokens"][:2] == ("less", "permeable")
    assert row["target_tokens"][-2:] == ("wooden", "vessel")
    assert row["fringe_trace"] == {"actual_pairs": 4, "termination": "outer_mismatch", "mismatch_pair": 5, "left": "p", "right": "e"}
    assert all(v for k, v in row["central_admission"].items() if k != "exact_letter_palindrome")


def test_compound_and_concessive_cuts_are_not_counted_without_live_traversal(report):
    result = report["screen"]
    shifted = [r for r in result["reviews"] if r["source_tokens"] != r["target_tokens"]]
    assert shifted
    assert any(r["source_tokens"][0] == "nonetheless" for r in shifted)
    assert all(r["live_boundary_witness"]["same_letter_tape"] for r in shifted)
    assert not any(r["live_boundary_witness"]["crossed_left_disagreements"] and r["live_boundary_witness"]["crossed_right_disagreements"] for r in shifted)
    assert result["reached_eleven_pair_derivations"] == result["live_two_sided_boundary_derivations_at_eleven_pairs"] == 0
    assert result["qualified_channels"] == 0 and result["channels"] == []


def test_all_renderings_screened_before_qualification_including_repeated_less(report):
    result = report["screen"]
    assert all(r["central_admission"]["length_band"] and r["central_admission"]["lexicon_words"] for r in result["reviews"])
    repeated = [r for r in result["reviews"] if not r["central_admission"]["distinct_words"]]
    assert len(repeated) == 5
    assert all(r["target_tokens"].count("less") == 2 for r in repeated)
    assert all(not r["eligible_for_external_provenance"] for r in repeated)


def test_derivation_multiplicity_is_not_unique_surface_count():
    p = next(productions())
    result = discover([p] * 3)
    assert result["source_target_derivation_pairs"] == 3
    assert result["distinct_target_surfaces"] == result["distinct_source_analyses"] == 1


@pytest.mark.parametrize("kwargs", [{"minimum_pairs": 10}, {"minimum_letters": 5}])
def test_eleven_pair_six_letter_gate_cannot_be_relaxed(kwargs):
    with pytest.raises(ValueError):
        discover([], **kwargs)


def test_geometric_eleven_pair_fixture_cannot_bypass_central_or_semantic_admission():
    source = ("qwe", "rtyuiopzb", "uv", "ubzpoi", "uytrewq")
    target = ("qwertyuiopzb", "uv", "ubzpoiuytrewq")
    result = discover([{"source_tokens": source, "target_tokens": target, "source_relation": {"fixture": True}}])
    row, = result["closure_reviews"]
    assert row["fringe_trace"]["actual_pairs"] >= 11
    assert row["live_boundary_witness"]["crossed_left_disagreements"] and row["live_boundary_witness"]["crossed_right_disagreements"]
    assert not row["central_admission"]["lexicon_words"]
    assert row["source_parse"] == row["independent_final_parse"] == []
    assert result["qualified_channels"] == 0


@pytest.mark.parametrize("middle,parity", [("uv", "odd"), ("uvv", "even")])
def test_all_exact_fixture_closures_are_rendered_and_rejected(middle, parity):
    words = ("qwertyb", middle, "ubytrewq")
    result = discover([{"source_tokens": words, "target_tokens": words, "source_relation": {"fixture": True}}])
    row, = result["closure_reviews"]
    assert row["fringe_trace"]["center_kind"] == parity
    assert row["rendered_diagnostic"] == " ".join(words).capitalize() + "."
    assert "independent_ordered_scalar_semantics" in row["failures"]
    assert row["external_provenance"] == "not_checked"
    assert not row["eligible_for_external_provenance"] and not row["promoted"]


@pytest.mark.parametrize("words,check", [
    (("abc", "dog", "god", "cba"), "no_self_palindromic_proper_multiword_span"),
    (("abc", "level", "cba"), "no_self_palindromic_word"),
    (("dogs", "dogs"), "distinct_words"),
])
def test_shortcut_rejections_are_kept(words, check):
    row = review({"source_tokens": words, "target_tokens": words, "source_relation": {"fixture": True}})
    assert not row["central_admission"][check]
    assert not row["promoted"]


def test_no_product_transport_or_originality_claim_on_zero_screen(report):
    result = report["screen"]
    assert result["exact_closure_derivations"] == result["distinct_exact_closure_surfaces"] == 0
    assert not result["full_palindrome_product_compiled"] and not result["transport_implemented"]
    assert result["model_queries_executed"] == 0
    assert report["provenance"]["external_candidate_provenance_searches"] == 0
    assert report["provenance"]["lexical_validation_requests"] == 1
    assert not report["provenance"]["originality_claim"]
    assert "cleft" in report["next_construction"]
