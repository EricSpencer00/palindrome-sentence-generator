from __future__ import annotations

from tools.audit_candidate_readability_20260915 import audit, audit_row, iter_rows


def test_iter_rows_preserves_rendered_text_and_source() -> None:
    rows = list(iter_rows({"rendered_probes": [{"rendered": "A man."}]}, "run.json"))
    assert rows == [{"source_run": "run.json", "rendered": "A man."}]
    recursive = list(iter_rows({"candidates": [{"rendered": "A baker repairs a gate."}]}, "recursive.json"))
    assert recursive == [{"source_run": "recursive.json", "rendered": "A baker repairs a gate."}]


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


def test_iter_rows_normalizes_authored_sentence_boundaries() -> None:
    rows = list(iter_rows({"probes": [{"left": "A calm nurse.", "right": "writes notes!"}]}, "run.json"))
    assert rows[0]["rendered"] == "A calm nurse. writes notes."


def test_audit_exposes_route_summary_as_diagnostic_only(tmp_path) -> None:
    run = tmp_path / "run.json"
    run.write_text('{"method":"test route","rendered_probes":[{"rendered":"A man."}]}')
    report = audit([run], seed=3, shuffles=2)
    assert report["route_summary"][0]["source_run"] == str(run)
    assert report["route_summary"][0]["rows"] == 1
    assert report["status"] == "diagnostic_not_human_readability_result"


def test_iter_rows_includes_list_valued_repair_phase() -> None:
    rows = list(iter_rows({"candidates": [{"rendered": "A man."}],
                           "repair": [{"rendered": "A nurse."}]}, "run.json"))
    assert [row["rendered"] for row in rows] == ["A man.", "A nurse."]


def test_iter_rows_includes_single_captured_authoring_probe() -> None:
    rows = list(iter_rows({"experiment": "probe", "captured_output": {"text": "A rare aura."}}, "run.json"))
    assert rows == [{"source_run": "run.json", "text": "A rare aura.", "rendered": "A rare aura.", "provenance": "probe"}]


def test_iter_rows_includes_rejected_rendered_repair_rows() -> None:
    rows = list(iter_rows({"experiment": "residual repair", "rejected": [
        {"text": "The nurse maps a bay. yab as pam esrun eht.", "provenance": "held-out residual"}
    ]}, "repair.json"))
    assert rows[0]["rendered"].startswith("The nurse maps")
    assert rows[0]["provenance"] == "held-out residual"
