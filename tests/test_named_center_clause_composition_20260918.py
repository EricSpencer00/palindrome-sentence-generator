import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/named_center_clause_composition_20260918.py"
RUN = ROOT / "runs/named-center-clause-composition-20260918.json"

def test_named_center_lane_records_independent_audit_and_no_shortcut():
    subprocess.run([sys.executable, str(SCRIPT)], check=True)
    data = json.loads(RUN.read_text())
    assert data["summary"]["exact_count"] == 0
    assert data["summary"]["reader_eligible_count"] == 0
    assert data["candidate_count"] == 125
    for row in data["candidates"]:
        assert row["audit"]["independent_two_pointer"] is False
        assert row["audit"]["forward_reverse_sha256"][0] != row["audit"]["forward_reverse_sha256"][1]
        assert row["provenance"]["catalogue_used"] is False
        assert row["novelty_preflight"]["punctuation_carries_letters"] is False
        assert row["novelty_preflight"]["complete_left_clause"]
        assert row["novelty_preflight"]["complete_right_clause"]
