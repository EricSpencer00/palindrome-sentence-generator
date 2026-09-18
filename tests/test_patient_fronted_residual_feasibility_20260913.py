import inspect
from pathlib import Path

import pytest

from experiments.patient_fronted_source_parser_20260913 import parse_source
from experiments.patient_fronted_final_parser_20260913 import parse_final, render_final
from experiments.patient_fronted_residual_feasibility_20260913 import productions, review, discover, run


@pytest.fixture(scope="module")
def report():
    return run()


def test_both_real_clause_orders_and_joint_patient_application_types():
    paths = list(productions())
    assert {p["source_construction"] for p in paths} == {"passive", "locative"}
    assert {p["joint_semantics"]["patient_number"] for p in paths} == {"singular", "plural"}
    assert {p["joint_semantics"]["defect"] for p in paths} == {"cracks", "tears"}
    assert len({p["joint_semantics"]["medium"] for p in paths}) == 4
    source = inspect.getsource(productions)
    assert "[::-1]" not in source and "reversed(" not in source


def test_source_and_final_grammars_are_separate_declarations():
    for name in ("patient_fronted_source_parser_20260913.py", "patient_fronted_final_parser_20260913.py"):
        code = Path("experiments", name).read_text()
        assert "from experiments" not in code
        assert "productions(" not in code and "joint_semantics" not in code
    path = next(p for p in productions() if p["source_tokens"] != p["target_tokens"])
    assert parse_source(path["source_tokens"]) and parse_final(path["target_tokens"])
    assert not parse_source(path["target_tokens"])
    assert not parse_final(path["source_tokens"])


def test_every_event_and_property_bind_to_the_same_patient(report):
    for row in report["screen"]["reviews"]:
        source, = row["source_parse"]
        final, = row["independent_final_parse"]
        assert source["semantic_relation_valid"] and final["semantic_relation_valid"]
        assert source["patient"]["id"] == source["repair"]["patient_id"] == source["adhesive_application"]["target_id"]
        assert final["patient"]["id"] == final["repair"]["patient_id"] == final["medium_application"]["target_id"]
        assert final["patient"]["id"] == final["reference"]["referent_id"] == final["locative_defect_bearer"]
        assert source["reference"]["number_matches"] and final["reference"]["number_matches"]
        assert final["repair"]["predicted_defect"] < final["repair"]["initial_defect"]


@pytest.mark.parametrize("old,new", [("it", "them"), ("is", "are"), ("cracks", "tears"), ("epoxy", "glue")])
def test_independent_parsers_reject_number_defect_and_material_mismatch(old, new):
    path = next(p for p in productions() if p["source_tokens"][:2] == ("red", "metalwork")
                and p["source_construction"] == "passive")
    a = tuple(new if w == old else w for w in path["source_tokens"])
    b = tuple(new if w == old else w for w in path["target_tokens"])
    assert not parse_source(a)
    assert not parse_final(b)


def test_unsupported_tool_missing_article_and_trailing_words_rejected():
    path = next(p for p in productions() if p["target_tokens"][-2:] == ("a", "spreader"))
    for words, parser in ((path["source_tokens"], parse_source), (path["target_tokens"], parse_final)):
        assert not parser(words[:-1] + ("hammer",))
        assert not parser(words[:-2] + ("spreader",))
        assert not parser(words + ("again",))


def test_locative_render_places_comma_after_full_patient_source_phrase():
    path = next(p for p in productions() if p["source_construction"] == "locative")
    rendered = render_final(path["target_tokens"])
    assert "regional museums, local" in rendered
    assert "".join(c for c in rendered.lower() if c.isalpha()) == "".join(path["target_tokens"])


def test_complete_finite_accounting_and_literal_pair_trace(report):
    paths = list(productions())
    result = report["screen"]
    assert len(paths) == result["source_target_derivation_pairs"] == 210
    assert len({p["target_tokens"] for p in paths}) == result["distinct_target_surfaces"] == 210
    assert len({p["source_tokens"] for p in paths}) == result["distinct_source_analyses"] == 210
    assert len(result["reviews"]) == 210 and result["finite_exhausted"]
    assert result["actual_pair_depth_distribution"] == {0: 171, 1: 6, 2: 27, 3: 3, 4: 3}
    for row in result["reviews"]:
        tape = row["normalized"]
        depth = row["fringe_trace"]["actual_pairs"]
        assert all(tape[i] == tape[-i - 1] for i in range(depth))
        assert tape[depth] != tape[-depth - 1]
        assert row["fringe_trace"]["mismatch_pair"] == depth + 1
    assert sum(r["derivation_pairs"] for r in result["outer_mismatches"]) == 210


def test_deepest_production_paths_are_four_pairs_not_eleven(report):
    result = report["screen"]
    deepest = [r for r in result["reviews"] if r["fringe_trace"]["actual_pairs"] == 4]
    assert len(deepest) == 3
    assert all(r["fringe_trace"]["left"] == "c" and r["fringe_trace"]["right"] == "e" for r in deepest)
    assert result["reached_eleven_pair_derivations"] == 0
    assert result["live_two_sided_boundary_derivations_at_eleven_pairs"] == 0
    assert result["qualified_channels"] == 0 and result["channels"] == []


def test_unvisited_compound_boundaries_are_never_counted(report):
    shifted = [r for r in report["screen"]["reviews"] if r["source_tokens"] != r["target_tokens"]]
    assert shifted
    for row in shifted:
        b = row["live_boundary_witness"]
        assert b["same_letter_tape"] and b["all_boundary_disagreements"]
        assert not (b["crossed_left_disagreements"] and b["crossed_right_disagreements"])


def test_central_lexicon_and_length_failures_are_preserved_not_relaxed(report):
    result = report["screen"]
    assert result["failure_counts"] == {"exact_letter_palindrome": 210, "length_band": 61, "lexicon_words": 90}
    epoxy_rows = [r for r in result["reviews"] if "epoxy" in r["target_tokens"]]
    assert len(epoxy_rows) == 90
    assert all(not r["central_admission"]["lexicon_words"] for r in epoxy_rows)
    assert all(not r["promoted"] for r in result["reviews"])


def test_derivation_multiplicity_is_not_surface_count():
    path = next(productions())
    result = discover([path, path])
    assert result["source_target_derivation_pairs"] == 2
    assert result["distinct_target_surfaces"] == result["distinct_source_analyses"] == 1


@pytest.mark.parametrize("kwargs", [{"minimum_pairs": 10}, {"minimum_letters": 5}])
def test_eleven_pair_and_six_letter_gate_cannot_be_relaxed(kwargs):
    with pytest.raises(ValueError):
        discover([], **kwargs)


def test_eleven_pair_two_boundary_fixture_cannot_substitute_for_english():
    source = ("qwe", "rtyuiopzb", "uv", "ubzpoi", "uytrewq")
    target = ("qwertyuiopzb", "uv", "ubzpoiuytrewq")
    result = discover([{"source_tokens": source, "target_tokens": target, "source_construction": "fixture"}])
    row, = result["closure_reviews"]
    assert row["central_admission"]["exact_letter_palindrome"]
    assert row["fringe_trace"]["actual_pairs"] >= 11
    assert row["live_boundary_witness"]["crossed_left_disagreements"]
    assert row["live_boundary_witness"]["crossed_right_disagreements"]
    assert row["source_parse"] == row["independent_final_parse"] == []
    assert not row["central_admission"]["lexicon_words"]
    assert result["qualified_channels"] == 0
    assert not row["eligible_for_external_provenance"]


@pytest.mark.parametrize("middle,parity", [("uv", "odd"), ("uvv", "even")])
def test_free_center_fixture_closures_are_all_rendered_and_rejected(middle, parity):
    tokens = ("qwertyb", middle, "ubytrewq")
    result = discover([{"source_tokens": tokens, "target_tokens": tokens, "source_construction": "fixture"}])
    row, = result["closure_reviews"]
    assert row["fringe_trace"]["center_kind"] == parity
    assert row["rendered_diagnostic"] == " ".join(tokens).capitalize() + "."
    assert "independent_source_and_final_semantics" in row["failures"]
    assert row["external_provenance"] == "not_checked"
    assert not row["eligible_for_external_provenance"] and not row["promoted"]


@pytest.mark.parametrize("tokens,check", [
    (("abc", "dog", "god", "cba"), "no_self_palindromic_proper_multiword_span"),
    (("abc", "level", "cba"), "no_self_palindromic_word"),
    (("dogs", "dogs"), "distinct_words"),
])
def test_every_shortcut_check_is_retained(tokens, check):
    row = review({"source_tokens": tokens, "target_tokens": tokens, "source_construction": "fixture"})
    assert not row["central_admission"][check]
    assert not row["promoted"]


def test_no_product_authoring_or_provenance_claim_on_unqualified_inventory(report):
    result = report["screen"]
    assert result["exact_closure_derivations"] == result["distinct_exact_closure_surfaces"] == 0
    assert result["closure_reviews"] == []
    assert not result["full_palindrome_product_compiled"]
    assert not result["authoring_interface_implemented"]
    assert result["model_queries_executed"] == 0
    assert report["provenance"]["external_searches_executed"] == 0
    assert report["provenance"]["external_review_required_before_promotion"]
    assert "result-state" in report["next_construction"]
