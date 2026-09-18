"""Exactness and search completeness on tiny fully enumerable fixture spaces."""
import itertools
import json

import pytest

from experiments.coupled_projection_repair_20260912 import (
    audit_artifact, independent_audit, ordinary_short_forms, record, search,
    source_projection, source_tape,
)


def test_shifted_word_boundaries_and_centre_crossing_word():
    rows, stats = search("ab c de", {"ab", "cba"}, beam=500, extra_edits=0)
    assert [row["text"] for row in rows] == ["Ab cba."]
    assert stats["unconstrained_minimum_substitutions"] == 2
    assert independent_audit(rows[0]["text"])["exact"]


def test_search_matches_brute_force_on_a_small_dictionary():
    vocabulary = {"ab", "cba", "abc", "ba", "ed", "cde", "edc", "de"}
    seed = "ab c de"
    expected = set()
    for count in range(1, 4):
        for words in itertools.product(sorted(vocabulary), repeat=count):
            tape = "".join(words)
            if len(tape) != 5 or len(words) != len(set(words)) or tape != tape[::-1]:
                continue
            if sum(a != b for a, b in zip(tape, "abcde")) <= 2:
                expected.add(" ".join(words).capitalize() + ".")
    rows, stats = search(seed, vocabulary, beam=10000, extra_edits=0)
    assert {row["text"] for row in rows} == expected
    assert stats["states_pruned_by_beam"] == 0
    assert expected


def test_unlexicalized_projection_is_exact_but_not_admitted():
    text = source_projection("Plain English sentence.", "left")
    assert independent_audit(text)["exact"]
    row = record(text, "fixture", "Plain English sentence.", "unlexicalized_left_projection")
    assert row["rejection_codes"]
    assert "length_band" in row["rejection_codes"]
    assert row["checks"]["source_length_preserved"]


def test_independent_audit_rejects_non_ascii_and_non_palindromes():
    assert independent_audit("Never odd or even.")["exact"]
    assert not independent_audit("Ordinary English.")["exact"]
    assert not independent_audit("éabaé")["exact"]
    assert source_tape("A-b C!") == "abc"


def test_extra_edit_budget_is_checked_at_odd_centre():
    # End pairs preserve source; the dictionary forces the central x -> c edit.
    rows, _ = search("abxba", {"ab", "cba"}, beam=100, extra_edits=0)
    assert rows == []
    rows, _ = search("abxba", {"ab", "cba"}, beam=100, extra_edits=1)
    assert [row["text"] for row in rows] == ["Ab cba."]
    assert rows[0]["extra_substitutions"] == 1


def test_invalid_budget_fails_early():
    with pytest.raises(ValueError):
        search("abc", {"ab"}, beam=0)


def test_short_form_policy_is_separate_from_dictionary_exactness():
    assert ordinary_short_forms("Draw nurses beside wide windows.")
    assert ordinary_short_forms("Go to my garden.")
    assert not ordinary_short_forms("Draw la tips oh diet nurse st net.")


def test_artifact_audit_does_not_trust_saved_flags(tmp_path):
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps({"records": [{
        "source_id": "fixture", "kind": "fixture", "text": "Draw la tips.",
        "checks": {"exact_letter_palindrome": True}, "rejection_codes": [],
    }]}))
    audit = audit_artifact(path)
    row = audit["records"][0]
    assert not row["independent_audit"]["exact"]
    assert row["excluded_short_forms"] == ["la"]
    assert "exact_letter_palindrome" in row["rejection_codes"]
    assert "experiment_ordinary_short_forms" in row["rejection_codes"]
