from __future__ import annotations

from tools.audit_candidate_readability_20260915 import audit_row, iter_rows


def test_iter_rows_preserves_rendered_text_and_source() -> None:
    rows = list(iter_rows({"rendered_probes": [{"rendered": "A man."}]}, "run.json"))
    assert rows == [{"source_run": "run.json", "rendered": "A man."}]


def test_audit_row_reports_exactness_and_gate_failures() -> None:
    row = audit_row(
        {"source_run": "run.json", "rendered": "An aide rips nine memos; some men inspire Diana."},
        seed=7,
        shuffles=2,
    )
    assert row["exact_letter_palindrome"] is True
    assert row["independent_sha256_exact"] is True
    assert row["letters"] == 38
    assert "length_band" in row["failed_checks"]
    assert row["reader_next_test"].startswith("Not reader-eligible")
