from __future__ import annotations

from experiments.bidirectional_typed_clause_transducer_20260914 import audit, run


def test_audit_uses_independent_ascii_tape() -> None:
    row = audit("A man, a plan.")
    assert row["normalized"] == row["independent_normalized"]
    assert row["independent_exact_audit"] is False
    assert row["reader_status"].startswith("human-unreviewed")


def test_transducer_run_declares_joint_typed_matching() -> None:
    result = run(state_limit=100)
    assert result["config"]["typed_slot_joint_choice"] is True
    assert result["config"]["independent_normalization"] is True
    assert result["provenance"]["material"].startswith("authored")

