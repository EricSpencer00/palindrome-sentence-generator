import inspect
from pathlib import Path

import pytest

from experiments.result_state_source_parser_20260913 import parse_source
from experiments.result_state_final_parser_20260913 import parse_final, render_final
from experiments.agentless_result_state_feasibility_20260913 import productions, review, discover, run


@pytest.fixture(scope="module")
def report():
    return run()


def test_joint_source_has_four_physical_result_relations_and_no_tape_reflection():
    paths = list(productions())
    assert {p["source_relation"]["repair"] for p in paths} == {"polished", "sealed", "tightened", "sanded"}
    assert {p["source_relation"]["material"] for p in paths} == {"metal", "wood", "glass"}
    assert len({p["target_tokens"][p["target_tokens"].index("now") + 1:] for p in paths}) == 8
    code = inspect.getsource(productions)
    assert "[::-1]" not in code and "reversed(" not in code


def test_source_and_final_grammars_are_independently_declared():
    for name in ("result_state_source_parser_20260913.py", "result_state_final_parser_20260913.py"):
        text = Path("experiments", name).read_text()
        assert "from experiments" not in text
        assert "productions(" not in text and "source_relation" not in text
    path = next(productions())
    assert parse_source(path["source_tokens"]) and parse_final(path["target_tokens"])
    assert not parse_source(path["target_tokens"]) and not parse_final(path["source_tokens"])


def test_all_result_states_follow_from_repair_and_bind_same_patient(report):
    for row in report["screen"]["reviews"]:
        source, = row["source_parse"]
        final, = row["independent_final_parse"]
        assert source["semantic_relation_valid"] and final["semantic_relation_valid"]
        assert source["patient_id"] == source["repair"]["patient_id"] == source["result"]["patient_id"]
        assert final["patient"]["id"] == final["initial_property"]["patient_id"]
        assert final["patient"]["id"] == final["participial_event"]["controlled_patient_id"] == final["measured_result"]["patient_id"]
        assert final["causal_witness"] == {"same_patient": True, "same_property": True, "material_support": True, "result_follows": True}
        assert source["predicted_state"] == final["predicted_state"]
        assert final["measured_result"]["satisfied"]


def test_counterfactual_repair_direction_is_rejected_not_syntax_membership():
    path = next(p for p in productions() if p["source_relation"]["repair"] == "tightened")
    for key, parser in (("source_tokens", parse_source), ("target_tokens", parse_final)):
        changed = tuple("loosened" if w == "tightened" else w for w in path[key])
        parsed, = parser(changed)
        assert not parsed["semantic_relation_valid"]
        assert parsed["predicted_state"] == 0


def test_same_numeric_direction_on_wrong_property_still_fails_causal_check():
    path = next(p for p in productions() if p["source_relation"]["repair"] == "sealed")
    for key, parser in (("source_tokens", parse_source), ("target_tokens", parse_final)):
        words = path[key]
        changed = words[:words.index("now") + 1] + ("lower",)
        parsed, = parser(changed)
        assert parsed["predicted_state"] == 0
        assert not parsed["semantic_relation_valid"]
        if key == "target_tokens":
            assert parsed["measured_result"]["satisfied"]  # 0 < 2, but this is NOT ridge height.
            assert not parsed["causal_witness"]["same_property"]
            assert not parsed["causal_witness"]["result_follows"]


def test_material_incompatible_counterfactual_is_not_accepted():
    path = next(p for p in productions() if p["source_relation"]["repair"] == "tightened"
                and p["source_relation"]["material"] == "metal")
    a = tuple("glasswork" if w == "metalwork" else w for w in path["source_tokens"])
    b = tuple("glass" if w == "metal" else w for w in path["target_tokens"])
    assert not parse_source(a)[0]["semantic_relation_valid"]
    final, = parse_final(b)
    assert not final["semantic_relation_valid"] and not final["causal_witness"]["material_support"]


def test_number_trailing_sentence_and_agent_insertion_are_rejected():
    path = next(productions())
    for key, parser in (("source_tokens", parse_source), ("target_tokens", parse_final)):
        words = path[key]
        assert not parser(tuple("are" if w == "is" else w for w in words))
        assert not parser(words + ("again",))
        cut = words.index("is")
        assert not parser(words[:cut] + ("by", "workers") + words[cut:])


def test_participial_rendering_preserves_every_letter():
    path = next(productions())
    rendered = render_final(path["target_tokens"])
    assert "museums, polished" in rendered and "pressure, is now" in rendered
    assert "".join(c for c in rendered.lower() if c.isalpha()) == "".join(path["target_tokens"])


def test_final_accounting_matches_literal_finite_paths_and_all_central_reviews(report):
    paths = list(productions())
    result = report["screen"]
    assert len(paths) == result["source_target_derivation_pairs"] == 30
    assert len({p["target_tokens"] for p in paths}) == result["distinct_target_surfaces"] == 30
    assert len({p["source_tokens"] for p in paths}) == result["distinct_source_analyses"] == 30
    assert len(result["reviews"]) == 30 and result["finite_exhausted"]
    assert result["failure_counts"] == {"exact_letter_palindrome": 30, "length_band": 3}
    assert all(row["central_admission"]["lexicon_words"] for row in result["reviews"])
    assert sum(m["derivation_pairs"] for m in result["outer_mismatches"]) == 30


def test_actual_deepest_fringe_dies_at_pair_five_and_is_not_an_eleven_pair_channel(report):
    result = report["screen"]
    assert result["actual_pair_depth_distribution"] == {0: 22, 1: 4, 4: 4}
    deepest = [r for r in result["reviews"] if r["fringe_trace"]["actual_pairs"] == 4]
    assert len(deepest) == 4
    for row in deepest:
        tape = row["normalized"]
        assert tape[:4] == tape[-4:][::-1]
        assert row["fringe_trace"] == {"actual_pairs": 4, "termination": "outer_mismatch", "mismatch_pair": 5, "left": "r", "right": "l"}
    assert result["reached_eleven_pair_derivations"] == result["live_two_sided_boundary_derivations_at_eleven_pairs"] == 0
    assert result["qualified_channels"] == 0 and result["channels"] == []


def test_boundary_changes_not_traversed_are_not_counted(report):
    rows = [r for r in report["screen"]["reviews"] if r["source_tokens"] != r["target_tokens"]]
    assert rows
    for row in rows:
        assert row["live_boundary_witness"]["same_letter_tape"]
        assert row["live_boundary_witness"]["all_boundary_disagreements"]
        assert not (row["live_boundary_witness"]["crossed_left_disagreements"] and row["live_boundary_witness"]["crossed_right_disagreements"])


def test_duplicate_derivations_do_not_inflate_unique_surface_count():
    path = next(productions())
    result = discover([path] * 3)
    assert result["source_target_derivation_pairs"] == 3
    assert result["distinct_target_surfaces"] == result["distinct_source_analyses"] == 1


@pytest.mark.parametrize("kwargs", [{"minimum_pairs": 10}, {"minimum_letters": 5}])
def test_eleven_pair_six_distinct_letter_gate_cannot_be_lowered(kwargs):
    with pytest.raises(ValueError):
        discover([], **kwargs)


def test_eleven_pair_two_cut_fixture_cannot_qualify_without_central_and_causal_admission():
    source = ("qwe", "rtyuiopzb", "uv", "ubzpoi", "uytrewq")
    target = ("qwertyuiopzb", "uv", "ubzpoiuytrewq")
    result = discover([{"source_tokens": source, "target_tokens": target, "source_relation": {"fixture": True}}])
    row, = result["closure_reviews"]
    assert row["central_admission"]["exact_letter_palindrome"]
    assert row["fringe_trace"]["actual_pairs"] >= 11
    assert row["live_boundary_witness"]["crossed_left_disagreements"] and row["live_boundary_witness"]["crossed_right_disagreements"]
    assert not row["central_admission"]["lexicon_words"]
    assert row["source_parse"] == row["independent_final_parse"] == []
    assert result["qualified_channels"] == 0


@pytest.mark.parametrize("middle,parity", [("uv", "odd"), ("uvv", "even")])
def test_free_midpoint_closures_rendered_and_preserved_even_when_rejected(middle, parity):
    words = ("qwertyb", middle, "ubytrewq")
    result = discover([{"source_tokens": words, "target_tokens": words, "source_relation": {"fixture": True}}])
    row, = result["closure_reviews"]
    assert row["fringe_trace"]["center_kind"] == parity
    assert row["rendered_diagnostic"] == " ".join(words).capitalize() + "."
    assert "independent_causal_semantics" in row["failures"]
    assert row["external_provenance"] == "not_checked"
    assert not row["eligible_for_external_provenance"] and not row["promoted"]


@pytest.mark.parametrize("words,check", [
    (("abc", "dog", "god", "cba"), "no_self_palindromic_proper_multiword_span"),
    (("abc", "level", "cba"), "no_self_palindromic_word"),
    (("dogs", "dogs"), "distinct_words"),
])
def test_central_shortcut_checks_are_never_skipped(words, check):
    row = review({"source_tokens": words, "target_tokens": words, "source_relation": {"fixture": True}})
    assert not row["central_admission"][check]
    assert check in row["failures"]
    assert not row["promoted"]


def test_no_full_product_interface_or_provenance_claim_when_screen_fails(report):
    result = report["screen"]
    assert result["exact_closure_derivations"] == result["distinct_exact_closure_surfaces"] == 0
    assert not result["full_palindrome_product_compiled"] and not result["authoring_interface_implemented"]
    assert result["model_queries_executed"] == 0 and result["promoted_candidates"] == []
    assert report["provenance"]["external_review_required_before_promotion"]
    assert report["provenance"]["external_searches_executed"] == 0
    assert "result-fronting" in report["next_construction"]
