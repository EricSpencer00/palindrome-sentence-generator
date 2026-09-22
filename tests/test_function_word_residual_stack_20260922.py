import hashlib
import json
from pathlib import Path

from experiments.function_word_residual_stack_20260922 import RESIDUALS
from experiments.productive_affix_return_stack_20260922 import complementary_boundary_mask
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "function_word_residual_stack_20260922.py"
RAW = ROOT / "artifacts" / "function-word-residual-stack-20260922" / "phrase-frontier.json"
RESULT = ROOT / "runs" / "function-word-residual-stack-20260922.json"


def _data():
    return json.loads(RESULT.read_text())


def test_remote_frontier_is_fixed_and_source_identical():
    raw = json.loads(RAW.read_text())
    result = _data()
    assert tuple(result["fixed_conditions"]["residuals"]) == RESIDUALS
    assert result["fixed_conditions"]["lexical_widening_after_run"] is False
    assert raw["provenance"]["host"] == "hst-bench"
    assert raw["provenance"]["python"] == "3.12.3"
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == raw["provenance"]["source_sha256"]
    assert hashlib.sha256(RAW.read_bytes()).hexdigest() == result["raw_frontier"]["sha256"]


def test_every_retained_phrase_cycle_satisfies_the_live_equation():
    raw = json.loads(RAW.read_text())
    assert [row["residual"] for row in raw["residual_frontiers"]] == list(RESIDUALS)
    for frontier in raw["residual_frontiers"]:
        residual = frontier["residual"]
        assert frontier["cycle_count"] == len(frontier["cycles"])
        for row in frontier["cycles"]:
            assert row["equation"]["holds"] is True
            assert row["x_tape"] + residual == residual + row["y_tape"][::-1]
            assert row["self_or_symmetric"] is False


def test_long_diagnostic_is_exact_lifo_but_rejected_before_acceptance():
    row = _data()["diagnostics"][1]
    tape = normalize_letters(row["rendered"])
    assert len(tape) == 48 > 44
    assert tape == tape[::-1]
    assert hashlib.sha256(tape.encode()).hexdigest() == row["sha256"]
    assert tokenize(row["rendered"]).count("on") == 1
    pushes = row["stack_pushes_outer_to_inner"]
    assert row["lifo_pop_order"] == [item["y"] for item in reversed(pushes)]
    for item in pushes:
        x = "".join(item["x"])
        y = "".join(item["y"])
        assert x + "on" == "on" + y[::-1]
    checks = mechanical_admission_checks(row["rendered"], min_letters=45, max_letters=160)
    assert checks["no_self_palindromic_proper_multiword_span"] is False
    left = ("trap", "at", "one", "most", "onset", "ones", "on")
    right = ("nose", "notes", "not", "some", "not", "a", "part")
    assert complementary_boundary_mask(left, right)["forbidden_internal"] == [4]


def test_per_residual_cursor_or_grammar_obstruction_is_exact():
    result = _data()
    raw = json.loads(RAW.read_text())
    raw_by_residual = {row["residual"]: row for row in raw["residual_frontiers"]}
    for row in result["per_residual_obstruction"]:
        frontier = raw_by_residual[row["residual"]]
        assert row["frontier_supported_y_tapes"] == frontier["stats"].get("frontier_supported_y_tapes", 0)
        assert row["equation_tape_hits"] == frontier["stats"].get("equation_tape_hits", 0)
        assert row["fresh_surface_cycles"] == frontier["cycle_count"]
        assert row["carrier_pairs"] == frontier["carrier_count"]
        assert row["obstruction"]
        assert "grammar_phase" in row
        assert "token_cursor" in row or "character_cursor" in row
    assert result["accepted_candidates"] == []
    assert result["reader_gate"]["status"] == "closed"
