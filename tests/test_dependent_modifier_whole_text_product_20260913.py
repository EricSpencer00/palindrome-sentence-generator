"""Tests for article--modifier dependent compilation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.whole_text_palindrome_product_20260913 import compile_slots, construct, exhaustive_reference
from experiments.dependent_modifier_whole_text_product_20260913 import (
    ARTIFACTS, ARTIFACT_ADJECTIVES, DERIVATIONS, PERSONS, PERSON_ADJECTIVES, PLACES, PLACE_ADJECTIVES,
    derive_svo, independent_parse, render, run,
)


def test_every_compiled_derivation_binds_article_to_immediate_modifier():
    assert len(DERIVATIONS) == 720
    for derivation in DERIVATIONS:
        words = tuple(slot[0].form for slot in derivation.choices)
        parsed = independent_parse(derivation, render(words))
        assert parsed["ok"], (derivation.identifier, parsed)
        if derivation.kind == "declarative_svo":
            assert derivation.choices[0][0].form == ("an" if derivation.choices[1][0].form[0] in "aeiou" else "a")
            assert derivation.choices[4][0].form == ("an" if derivation.choices[5][0].form[0] in "aeiou" else "a")
        else:
            assert derivation.choices[1][0].form == ("an" if derivation.choices[2][0].form[0] in "aeiou" else "a")
            assert derivation.choices[5][0].form == ("an" if derivation.choices[6][0].form[0] in "aeiou" else "a")


def test_independent_reparse_rejects_surface_article_modifier_mismatch():
    derivation = derive_svo(PERSONS[0], PERSON_ADJECTIVES[0], ARTIFACTS[0],
                            next(item for item in ARTIFACT_ADJECTIVES if item.form == "brief"))
    good = render(("an", "agile", "artist", "built", "a", "brief", "canvas"))
    bad = good.replace("An agile", "A agile")
    assert independent_parse(derivation, good)["ok"]
    assert not independent_parse(derivation, bad)["ok"]


def test_kernel_tiny_oracle_and_shifted_boundaries_remain_exact():
    slots = (("ij", "ix"), ("k",), ("ji", "zz"))
    result = construct(compile_slots(slots), max_states=1_000)
    assert {tuple(row["words"]) for row in result["records"]} == exhaustive_reference(slots) == {("ij", "k", "ji")}


def test_fresh_run_reports_all_derivations_exhausted_without_readability_claim():
    result = run(max_states=1_000)
    assert result["eligible_derivation_count"] == 720
    assert all(item["kernel"]["states_exhausted"] and not item["kernel"]["truncated"] for item in result["derivation_runs"])
    assert result["reader_facing_next_test"].startswith("Only an admitted closure")
    for row in result["exact_closures"]:
        assert row["independent_terminal_path_replay"]["ok"]
        assert row["independent_parse"]["ok"]
        assert row["independent_exact_audit"]["exact"]
