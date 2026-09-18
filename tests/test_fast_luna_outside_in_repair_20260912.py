from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.fast_luna_outside_in_repair_20260912 import (  # noqa: E402
    EXACT_NEAR_MISS,
    next_repair_operator,
    normalize,
    repair_once,
    run,
    strict_validation,
)


def test_independent_normalizer_and_exact_near_miss() -> None:
    tape = normalize(EXACT_NEAR_MISS)
    assert tape == tape[::-1]
    assert len(tape) == 30
    report = strict_validation(EXACT_NEAR_MISS, set())
    assert report["checks"]["exact_letter_palindrome"] is True
    assert report["checks"]["no_repeated_units"] is True
    assert report["checks"]["no_self_palindromic_units"] is True
    assert report["admitted"] is False


def test_repair_is_outside_in_and_reversible() -> None:
    trace = repair_once("The quiet nurse reads one warm letter.")
    assert trace["operator"] == "same_pos_outside_in_edit"
    assert trace["rendered"] is None or trace["mismatches_after"] < trace["mismatches_before"]
    two = next_repair_operator("The quiet nurse reads one warm letter.")
    assert two["operator"] == "bounded_two_word_same_pos_repair"
    assert two["rendered"] is None or two["mismatches_after"] < two["mismatches_before"]


def test_run_reports_rendered_audits_and_no_strict_shortcut_survivor() -> None:
    result = run()
    assert result["candidates"]
    assert all("rendered" in row and "normalized" in row and "length" in row for row in result["candidates"])
    assert not any(row["admitted"] for row in result["candidates"])
    assert "next_repair_operator" in result
    assert result["reader_facing_next_test"].startswith("Blind readers")
