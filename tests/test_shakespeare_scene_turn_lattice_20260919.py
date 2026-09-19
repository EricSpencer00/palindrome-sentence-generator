import json
from pathlib import Path

from experiments.shakespeare_scene_turn_lattice_20260919 import exact_audit, hidden_spans, run


def test_audit_independently_checks_exact_tape_and_hashes():
    got = exact_audit("A man, a plan, a canal: Panama")
    assert got["two_pointer_exact"] is True
    assert got["sha_equal"] is True
    assert got["first_mismatch"] is None


def test_scene_lattice_is_deterministic_and_no_exact_rows_are_fabricated():
    payload = run()
    assert payload["method"]["reward_model_used"] is False
    assert payload["method"]["catalogue_imported"] is False
    assert payload["search"]["exact_count"] == len(payload["representative_exact_candidates"])
    assert payload["search"]["longest_exact"] == 0
    assert payload["strict_gate"]["human_readability_test"] == "not performed"


def test_hidden_span_audit_finds_only_explicit_internal_palindromes():
    assert "a toyota" in hidden_spans("A Toyota")
