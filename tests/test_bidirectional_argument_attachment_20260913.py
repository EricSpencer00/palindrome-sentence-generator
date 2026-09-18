from pathlib import Path

import pytest

from experiments.bidirectional_argument_attachment_20260913 import (
    compile_bidirectional, frontier_certificate, review_closures, run,
)
from experiments.boundary_shifting_sentence_lattice_20260913 import search_lattice, toy_lattice
from experiments.credential_attachment_source_20260913 import Source, source_steps, source_meaning
from experiments.credential_attachment_target_20260913 import Target, target_steps, target_meaning
from experiments.credential_attachment_validator_20260913 import parse_credential_sentence, render_credential_sentence


def fixture_languages(middle="uv", letters="bdefgh"):
    source, target = [], []
    for letter in letters:
        source.append(("qwe", "rty" + letter, middle, "u" + letter + "yt", "rewq"))
        target.append(("qwerty" + letter, middle, "u" + letter + "ytrewq"))
    return toy_lattice(source), toy_lattice(target)


@pytest.mark.parametrize("middle,center", [("uv", "odd"), ("uvv", "even")])
def test_fixture_two_sided_frontier_six_paired_letters_and_free_centers(middle, center):
    source, target = fixture_languages(middle)
    lattice, accounting = compile_bidirectional(source, target, lambda s: "fixture", lambda s: "fixture")
    selected, certificate = frontier_certificate(lattice)
    channel, = certificate["channels"]
    assert channel["qualified"] and channel["paired_next_letters"] == list("bdefgh")
    assert all(w["actual_pairs"] == 6 and w["left_crossed_cuts"] == [3] and w["right_crossed_cuts"] == [4] for w in channel["witnesses"])
    result = search_lattice(selected, probe_pairs=6)
    assert len(result["records"]) == 6
    assert all(row["center_kind"] == center and row["tape"] == row["tape"][::-1] for row in result["records"])
    assert accounting["two_sided_boundary_filtered"]["accepted_source_target_derivation_pairs"] == 6
    assert accounting["two_sided_boundary_filtered"]["distinct_rendered_target_surfaces"] == 6


def test_fewer_than_six_continuations_does_not_qualify():
    source, target = fixture_languages(letters="bdefg")
    lattice, _ = compile_bidirectional(source, target, lambda s: "fixture", lambda s: "fixture")
    selected, certificate = frontier_certificate(lattice)
    assert certificate["qualified_channels"] == 0
    assert not selected.accepting
    assert not search_lattice(selected)["records"]


def test_missing_right_resegmentation_is_excluded_before_frontier():
    target = [("qwerty" + c, "uv", "u" + c + "ytrewq") for c in "bdefgh"]
    source = [("qwe", "rty" + c, "uv", "u" + c + "ytrewq") for c in "bdefgh"]
    lattice, counts = compile_bidirectional(toy_lattice(source), toy_lattice(target), lambda s: "fixture", lambda s: "fixture")
    assert counts["unrestricted"]["distinct_rendered_target_surfaces"] == 6
    assert counts["two_sided_boundary_filtered"]["distinct_rendered_target_surfaces"] == 0
    assert not lattice.accepting


def test_render_review_keeps_every_rejected_closure_and_repetition_check():
    words = ("qwertyb", "uv", "ubytrewq")
    tape = "".join(words)
    reviews = review_closures([{"words": words, "tape": tape, "matched_depth": 8}], min_letters=1)
    row, = reviews
    assert row["rendered_diagnostic"] == "Qwertyb uv ubytrewq."
    assert row["exact"] and row["central_admission"]["exact_letter_palindrome"]
    assert not row["central_admission"]["lexicon_words"]
    assert "independent_final_semantics" in row["failures"]
    assert not row["promoted"]
    repeated = review_closures([{"words": ("dogs", "dogs"), "tape": "dogsdogs", "matched_depth": 6}], min_letters=1)[0]
    assert not repeated["central_admission"]["distinct_words"]


@pytest.mark.parametrize("middle,check", [(("fg", "gf"), "no_self_palindromic_proper_multiword_span"), (("level",), "no_self_palindromic_word")])
def test_admission_rejects_multiword_and_single_content_islands(middle, check):
    words = ("qwertyb",) + middle + ("bytrewq",)
    row, = review_closures([{"words": words, "tape": "".join(words), "matched_depth": 7}], min_letters=1)
    assert not row["central_admission"][check]
    assert not row["promoted"]


def test_distinct_source_target_attachment_declarations_and_final_parser():
    for filename in ("credential_attachment_source_20260913.py", "credential_attachment_target_20260913.py", "credential_attachment_validator_20260913.py"):
        assert "from experiments" not in Path("experiments", filename).read_text()
    a = Source("complete", "identify", "guards", "door")
    b = Target("finished", predicate="identify", sponsor="guards", role_domain="door")
    assert source_meaning(a)["shared_core"] == target_meaning(b)["shared_core"]
    assert source_meaning(a)["attachment"]["attaches_to"] == "duty_domain"
    assert target_meaning(b)["attachment"]["attaches_to"] == "attendant"
    words = "key cards identify the guards doorman".split()
    assert parse_credential_sentence(words)[0]["semantic_relation_valid"]
    assert render_credential_sentence(words) == "Key cards identify the guards' doorman."
    assert not parse_credential_sentence("key cards eat the guards doorman".split())[0]["semantic_relation_valid"]
    assert not parse_credential_sentence("key cards identify the guards door man".split())


def test_derivation_multiplicity_is_not_called_distinct_surfaces():
    targets = [("qwertyb", "uv", "ubytrewq")]
    sources = [("qwe", "rtyb", "uv", "ubyt", "rewq"), ("qw", "ertyb", "uv", "uby", "trewq")]
    lattice, counts = compile_bidirectional(toy_lattice(sources), toy_lattice(targets), lambda s: "fixture", lambda s: "fixture")
    assert counts["two_sided_boundary_filtered"]["accepted_source_target_derivation_pairs"] == 2
    assert counts["two_sided_boundary_filtered"]["distinct_rendered_target_surfaces"] == 1


def test_actual_credential_inventory_remains_explicitly_unqualified():
    result = run()
    assert result["status"] == "construction_regime_not_qualified"
    assert result["production_frontier_certificate"]["deepest"]["depth"] == 0
    assert result["production_frontier_certificate"]["qualified_channels"] == 0
    assert result["requirements_unmet"]
    assert result["exact_closures"] == 0
    assert result["pending_external_review"] == result["promoted_candidates"] == []
