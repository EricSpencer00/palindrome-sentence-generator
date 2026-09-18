import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "syntax-constrained-bridge-composition-20260918.json"

def test_clause_bridge_lane_records_complete_clause_search_and_controls():
    subprocess.run(["python3", "experiments/syntax_constrained_bridge_composition_20260918.py"], cwd=ROOT, check=True)
    data = json.loads(RUN.read_text())
    assert data["experiment"] == "syntax-constrained-bridge-composition-20260918"
    assert data["summary"]["exact_count"] == 0
    assert len(data["controls"]) == 3
    assert all(not row["audit"]["exact"] for row in data["controls"])
    assert "complete authored clause" in data["method"]
    assert data["summary"]["next_repair"]
