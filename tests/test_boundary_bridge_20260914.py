import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.boundary_bridge_20260914 import SCENE, bridge, independent_audit, resume_from_witness, run

def test_bridge_keeps_complete_clauses_and_nonempty_residual():
    rows = bridge(*SCENE)
    assert all(row["residual"] for row in rows)
    assert all(row["left"].endswith(".") and row["right"].endswith(".") for row in rows)
    for row in rows:
        left, right = ''.join(c for c in row["left"].lower() if c.isalpha()), ''.join(c for c in row["right"].lower() if c.isalpha())[::-1]
        assert left.startswith(right[:row["cancelled_letters"]])

def test_run_reports_auditable_provenance_and_next_operator():
    result = run()
    assert result["provenance"].startswith("Fresh authored")
    assert result["program_sha256"]
    assert result["next_operator_if_empty"]
    for row in result["records"]:
        assert row["audit"] == independent_audit(row["text"])

def test_resume_witness_can_close_with_third_clause():
    witness = {"left": "abc d.", "right": "ba.", "residual": "d",
               "cancelled_letters": 1}
    assert resume_from_witness(witness, ("c.",), min_letters=7) == [(0,)]
    text = " ".join((witness["left"], "c.", witness["right"]))
    assert text == "abc d. c. ba."
    assert independent_audit(text)["exact"]
