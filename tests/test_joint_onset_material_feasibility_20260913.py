import inspect
from pathlib import Path

import pytest

from experiments.joint_material_repair_validator_20260913 import parse_sentence, render_sentence
from experiments.joint_onset_material_feasibility_20260913 import (
    boundary_witness, discover, productions, review, run,
)


@pytest.fixture(scope="module")
def report():
    return run()


def test_joint_source_varies_typed_events_materials_and_finishes():
    paths = list(productions())
    assert {p["source_semantics"]["first_action"] for p in paths} == {"sort", "solder", "seal", "brush"}
    assert {p["source_semantics"]["material"] for p in paths} == {"metal", "wood", "fabric"}
    assert {p["source_semantics"]["finish"] for p in paths} == {"high_gloss", "matte", "polished_hardwood"}
    assert all(p["source_semantics"]["material"] == "metal" for p in paths if p["source_semantics"]["first_action"] == "solder")
    assert all(p["source_semantics"]["material"] == "wood" for p in paths if p["source_semantics"]["finish"] == "polished_hardwood")
    source = inspect.getsource(productions)
    assert "[::-1]" not in source and "reversed(" not in source


def test_every_material_property_and_repair_bind_to_the_same_owned_artwork(report):
    for row in report["screen"]["reviews"]:
        for key in ("source_independent_parses", "target_independent_parses"):
            parsed, = row[key]
            assert parsed["semantic_relation_valid"]
            assert parsed["ownership"]["patient_id"] == parsed["first_event"]["patient_id"]
            assert parsed["ownership"]["patient_id"] == parsed["repair_event"]["patient_id"]
            assert parsed["ownership"]["patient_id"] == parsed["final_property"]["bearer_id"]
            assert parsed["repair_event"]["predicted_defect"] < parsed["repair_event"]["initial_defect"]
            assert parsed["repair_event"]["affected_property"] != parsed["first_event"]["affected_property"]
            assert parsed["final_property"]["material_compatible"]


def test_validator_does_not_import_source_grammar_or_accept_claimed_parse():
    code = Path("experiments/joint_material_repair_validator_20260913.py").read_text()
    assert "from experiments" not in code
    assert "productions(" not in code and "source_semantics" not in code
    path = next(productions())
    parsed = parse_sentence(path["target_tokens"])
    assert parsed and parsed[0]["semantic_relation_valid"]


def test_independent_parser_rejects_soldering_wood_and_mismatched_final_substrate():
    path = next(p for p in productions() if p["source_semantics"]["first_action"] == "solder")
    words = list(path["target_tokens"])
    words[words.index("metal")] = "oak"
    parsed, = parse_sentence(words)
    assert not parsed["first_event"]["material_compatible"]
    assert not parsed["semantic_relation_valid"]
    original = path["target_tokens"]
    finish = original.index("this") + 1
    wrong_substrate = original[:finish] + ("polished", "red", "hard", "wood", "art")
    parsed, = parse_sentence(wrong_substrate)
    assert not parsed["final_property"]["material_compatible"]
    assert not parsed["semantic_relation_valid"]


def test_ambiguous_reference_wrong_defect_and_trailing_material_rejected():
    words = next(productions())["target_tokens"]
    ambiguous = tuple("their" if w == "this" else w for w in words)
    assert not parse_sentence(ambiguous)[0]["semantic_relation_valid"]
    wrong_defect = tuple("tears" if w == "cracks" else w for w in words)
    assert not parse_sentence(wrong_defect)[0]["semantic_relation_valid"]
    assert not parse_sentence(words + ("metal",))


def test_actual_production_channel_reaches_ten_then_reports_exact_mismatch(report):
    result = report["screen"]
    assert result["actual_pair_depth_distribution"] == {0: 189, 4: 9, 6: 27, 7: 9, 8: 6, 9: 9, 10: 3}
    deepest = [r for r in result["reviews"] if r["fringe_trace"]["actual_pairs"] == 10]
    assert len(deepest) == 3
    for row in deepest:
        tape = row["normalized"]
        assert tape[:10] == tape[-10:][::-1]
        assert row["fringe_trace"] == {"actual_pairs": 10, "termination": "outer_mismatch", "mismatch_pair": 11, "left": "d", "right": "g"}
        assert all(v for k, v in row["central_admission"].items() if k != "exact_letter_palindrome")


def test_raw_two_sided_geometry_is_not_claimed_as_reached_production_shift(report):
    result = report["screen"]
    assert result["raw_two_sided_boundary_geometry_pairs"] == 18
    assert result["production_two_sided_crossings_after_ten_pairs"] == 0
    rows = [r for r in result["reviews"] if r["source_tokens"] != r["target_tokens"]]
    assert rows
    for row in rows:
        b = row["boundary_witness"]
        assert b["same_letter_tape"]
        assert b["all_boundary_disagreements"]
        assert not (b["crossed_left_disagreements"] and b["crossed_right_disagreements"])
    channel, = result["channels"]
    assert channel["source_target_derivation_pairs"] == 3
    assert channel["two_sided_boundary_eligible_derivations"] == 0
    assert channel["paired_next_letters"] == channel["unfiltered_paired_next_letters"] == []
    assert not channel["qualified"]


def test_distinct_boundary_mechanism_is_letter_preserving_but_fixture_only():
    source = ("qwe", "rtyuiopb", "uv", "ubpoi", "uytrewq")
    target = ("qwertyuiopb", "uv", "ubpoiuytrewq")
    witness = boundary_witness(source, target, 10)
    assert witness["same_letter_tape"]
    assert witness["crossed_left_disagreements"] == [3]
    assert witness["crossed_right_disagreements"] == [7]
    # Synthetic tokens cannot become eligible merely by satisfying cut geometry.
    row = review({"source_tokens": source, "target_tokens": target, "source_semantics": {"fixture": True}})
    assert not row["central_admission"]["lexicon_words"]
    assert not row["eligible_for_external_provenance"]


def test_exhaustive_derivation_and_surface_accounting_match_literal_paths(report):
    paths = list(productions())
    result = report["screen"]
    assert len(paths) == result["source_target_derivation_pairs"] == 252
    assert len({p["target_tokens"] for p in paths}) == result["distinct_rendered_target_surfaces"] == 252
    assert len({p["source_tokens"] for p in paths}) == result["distinct_source_analyses"] == 252
    assert len(result["reviews"]) == 252 and result["finite_exhausted"]
    assert sum(r["derivation_pairs"] for r in result["outer_mismatches"]) == 252
    assert result["failure_counts"] == {"distinct_words": 27, "exact_letter_palindrome": 252}
    assert all("".join(p["source_tokens"]) == "".join(p["target_tokens"]) for p in paths)


def test_duplicate_derivations_are_not_unique_surfaces():
    path = next(productions())
    result = discover([path, path])
    assert result["source_target_derivation_pairs"] == 2
    assert result["distinct_rendered_target_surfaces"] == 1
    assert result["distinct_source_analyses"] == 1


def test_different_source_tape_cannot_supply_boundary_certificate():
    path = next(p for p in productions() if p["target_tokens"][:2] == ("traders", "solder")
                and p["source_semantics"]["finish"] == "high_gloss")
    changed = {**path, "source_tokens": ("conservators",) + path["source_tokens"][1:]}
    result = discover([changed])
    row, = result["reviews"]
    assert "source_target_same_tape" in row["failures"]
    assert not row["eligible_for_external_provenance"]
    channel, = result["channels"]
    assert not channel["witnesses"][0]["safe_independent_semantic_analysis"]
    assert not channel["qualified"]


@pytest.mark.parametrize("kwargs", [{"pairs": 9}, {"letters": 5}])
def test_gate_cannot_be_relaxed(kwargs):
    with pytest.raises(ValueError):
        discover([], **kwargs)


@pytest.mark.parametrize("middle,parity", [("uv", "odd"), ("uvv", "even")])
def test_all_fixture_closures_rendered_reparsed_and_fail_closed(middle, parity):
    tokens = ("qwertyb", middle, "ubytrewq")
    result = discover([{"source_tokens": tokens, "target_tokens": tokens, "source_semantics": {"fixture": True}}])
    row, = result["closure_reviews"]
    assert result["exact_closure_derivations"] == result["distinct_exact_closure_surfaces"] == 1
    assert row["fringe_trace"]["center_kind"] == parity
    assert row["rendered_diagnostic"] == render_sentence(tokens)
    assert row["source_independent_parses"] == row["target_independent_parses"] == []
    assert not row["eligible_for_external_provenance"]
    assert row["external_provenance"] == "not_checked" and not row["originality_claim"]
    assert not row["promoted"]


@pytest.mark.parametrize("tokens,check", [
    (("abc", "dog", "god", "cba"), "no_self_palindromic_proper_multiword_span"),
    (("abc", "level", "cba"), "no_self_palindromic_word"),
    (("dogs", "dogs"), "distinct_words"),
])
def test_admission_shortcut_rejections_preserved(tokens, check):
    row = review({"source_tokens": tokens, "target_tokens": tokens, "source_semantics": {"fixture": True}})
    assert not row["central_admission"][check]
    assert check in row["failures"]
    assert not row["promoted"]


def test_unqualified_screen_compiles_neither_full_product_nor_transport(report):
    assert report["status"] == "finite_screen_unqualified"
    result = report["screen"]
    assert result["qualified_channels"] == result["distinct_exact_closure_surfaces"] == 0
    assert result["closure_reviews"] == []
    assert not result["full_palindrome_product_compiled"]
    assert not result["model_transport_implemented"] and result["model_queries_executed"] == 0
    assert result["promoted_candidates"] == []
    assert report["provenance"]["external_searches_executed"] == 0
    assert report["provenance"]["external_review_required_before_promotion"]
    assert "patient-fronted" in report["next_construction"]
