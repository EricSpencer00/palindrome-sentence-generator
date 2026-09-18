"""Tests for strict typed derivation compilation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.whole_text_palindrome_product_20260913 import compile_slots, construct, exhaustive_reference
from experiments.typed_derivation_whole_text_product_20260913 import (
    ARTIFACT_NOUNS, DERIVATIONS, PERSON_NOUNS, PLACE_NOUNS, derive_imperative,
    derive_svo, independent_parse, grammar_for, render, replay_words, run,
)


def test_derivations_are_role_typed_and_article_legal():
    assert len(DERIVATIONS) == len(PERSON_NOUNS) * len(ARTIFACT_NOUNS) + len(ARTIFACT_NOUNS) * len(PLACE_NOUNS)
    for derivation in DERIVATIONS:
        for role, choices in zip(derivation.roles, derivation.choices):
            assert choices, role
        for words in zip(*derivation.choices):
            # Every word tuple is not necessarily a complete derivation, but
            # no determiner choice itself can violate article phonology for a
            # fixed noun role.
            assert all(word.form.islower() for word in words)
        parsed = independent_parse(derivation, render(tuple(c[0].form for c in derivation.choices)))
        assert parsed["ok"], (derivation.identifier, parsed)


def test_independent_parser_rejects_bad_article_and_wrong_valency():
    derivation = derive_svo(PERSON_NOUNS[0], ARTIFACT_NOUNS[0])
    text = "An agile artist built a brief canvas."
    assert independent_parse(derivation, text)["ok"]
    bad_article = text.replace("An agile", "A agile")
    assert not independent_parse(derivation, bad_article)["ok"]
    bad_valency = text.replace("artist built", "artist is")
    assert not independent_parse(derivation, bad_valency)["ok"]


def test_terminal_path_replay_and_shifted_center_oracle():
    slots = (("ij", "ix"), ("k",), ("ji", "zz"))
    grammar = compile_slots(slots)
    expected = exhaustive_reference(slots)
    result = construct(grammar, max_states=1_000)
    observed = {tuple(row["words"]) for row in result["records"]}
    assert observed == expected == {("ij", "k", "ji")}
    assert result["states_exhausted"] and not result["truncated"]


def test_fresh_run_reports_exhaustion_and_no_readability_claim():
    result = run(max_states=1_000)
    assert result["eligible_derivation_count"] == 32
    assert all(item["kernel"]["states_exhausted"] for item in result["derivation_runs"])
    assert all(not item["kernel"]["truncated"] for item in result["derivation_runs"])
    assert result["reader_facing_next_test"].startswith("Only an admitted closure")
    for row in result["exact_closures"]:
        assert row["independent_path_replay"]["ok"]
        assert row["independent_parse"]["ok"]
        assert row["independent_exact_audit"]["exact"]

