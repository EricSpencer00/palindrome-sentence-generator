import pytest

from experiments.relative_coordination_endpoint_feasibility_20260913 import (
    discover, endpoint_witness, outer_anadrome_scaffold, parse_source, parse_target,
    run, source_paths, target_paths,
)


def test_complete_source_and_target_have_genuinely_different_attachments():
    source = "bookbinders who repair worn covers and printers inspect toolboxes".split()
    target = "book binders who repair worn covers and printers inspect tool boxes".split()
    a, b = parse_source(source), parse_target(target)
    assert "".join(source) == "".join(target)
    assert a["coordination_attaches_to"] == "relative_clause_objects"
    assert b["coordination_attaches_to"] == "main_clause_subjects"
    assert a["relative_clause"]["coordinated_objects"][-1]["type"] == "repairable_machine"
    assert b["main_agents"][-1]["type"] == "human_trade"
    assert len(a["main_agents"]) == 1 and len(b["main_agents"]) == 2


def test_all_enumerated_production_paths_parse_completely():
    assert all(parse_source(words) for words in source_paths())
    assert all(parse_target(words) for words in target_paths())


def test_independent_parsers_reject_wrong_segmentation_and_fragments():
    source = next(source_paths())
    target = next(target_paths())
    assert parse_source(target) is None
    assert parse_target(source) is None
    assert parse_source(source[:-1]) is None
    assert parse_target(target[:-1]) is None


def test_real_production_boundary_changes_are_not_misreported_as_matched_pairs():
    source = "bookbinders who repair covers and printers inspect toolboxes".split()
    target = "book binders who repair covers and printers inspect tool boxes".split()
    witness = endpoint_witness(source, target)
    assert witness["left_boundary_disagreements"] == [4]
    assert witness["right_boundary_disagreements"] == [5]
    assert witness["matched_pairs_capped_at_seven"] == 0
    assert not witness["six_actual_pairs_and_both_shifts"]


def test_six_continuations_must_be_paired_in_actual_complete_paths():
    # Non-English mechanism fixture only; never admitted as production data.
    source, target = [], []
    for letter in "bdefgh":
        source.append(("qwe", "rty" + letter, "uv", "u" + letter + "yt", "rewq"))
        target.append(("qwerty" + letter, "uv", "u" + letter + "ytrewq"))
    result = discover(source, target, lambda words: {"fixture": True}, lambda words: {"fixture": True})
    assert result["qualified_channels"] == 1
    assert result["qualified_target_surfaces"] == result["qualified_derivation_pairs"] == 6
    assert result["channels"][0]["paired_next_letters"] == list("bdefgh")
    # Same vocabulary, but permute the right-side letters so none pair.
    unpaired_source, unpaired_target = [], []
    for left, right in zip("bdefgh", "defghb"):
        unpaired_source.append(("qwe", "rty" + left, "uv", "u" + right + "yt", "rewq"))
        unpaired_target.append(("qwerty" + left, "uv", "u" + right + "ytrewq"))
    unpaired = discover(unpaired_source, unpaired_target, lambda words: {}, lambda words: {})
    assert unpaired["qualified_channels"] == 0
    assert len(unpaired["channels"][0]["left_next_letters"]) == 6
    assert len(unpaired["channels"][0]["right_next_letters"]) == 6
    assert unpaired["channels"][0]["paired_next_letters"] == []


def test_derivation_pair_and_surface_counts_are_separate():
    a = ("qwe", "rtyb", "uv", "ubyt", "rewq")
    b = ("q", "wertyb", "uv", "ubyt", "rewq")
    target = ("qwertyb", "uv", "ubytrewq")
    result = discover([a, b], [target], lambda words: {}, lambda words: {})
    assert result["same_tape_derivation_pairs"] == 2
    assert result["distinct_same_tape_target_surfaces"] == 1
    assert result["qualified_channels"] == 0


def test_known_tape_exclusion_and_no_gate_relaxation():
    a = ("qwe", "rtyb", "uv", "ubyt", "rewq")
    b = ("qwertyb", "uv", "ubytrewq")
    result = discover([a], [b], lambda words: {}, lambda words: {}, known={"".join(a)})
    assert result["same_tape_derivation_pairs"] == 0
    assert result["source_paths_rejected"] == result["target_paths_rejected"] == 1
    with pytest.raises(ValueError):
        discover([], [], lambda words: {}, lambda words: {}, pairs=5)
    with pytest.raises(ValueError):
        discover([], [], lambda words: {}, lambda words: {}, next_letters=5)


def test_generic_outer_reverse_filter_without_a_catalogue_fixture():
    assert outer_anadrome_scaffold(("qwertys", "uv", "ytrewq"))
    assert not outer_anadrome_scaffold(("qwertyb", "uv", "ubytrewq"))


def test_production_stops_before_any_full_product_if_unqualified():
    result = run()
    assert result["source_paths_enumerated"] == result["target_paths_enumerated"] == 960
    assert result["same_tape_derivation_pairs"] == result["distinct_same_tape_target_surfaces"] == 960
    assert result["finite_enumeration_complete"]
    assert len(result["enumerated_derivation_pairs"]) == 960
    assert sum(result["actual_matched_pair_distribution"].values()) == 960
    assert result["qualified_channels"] == result["qualified_target_surfaces"] == 0
    assert result["status"] == "endpoint_feasibility_unqualified"
    assert not result["full_palindrome_product_compiled"]
    assert result["candidate_count"] == 0
