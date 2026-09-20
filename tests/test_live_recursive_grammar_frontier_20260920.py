import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUN = ROOT / "runs/live-recursive-grammar-frontier-20260920.json"


def test_live_recursive_frontier_records_obligations_and_independent_audits():
    subprocess.run([sys.executable, str(ROOT / "experiments/live_recursive_grammar_frontier_20260920.py")], check=True)
    data = json.loads(RUN.read_text())
    assert data["novelty_preflight"]["status"] == "passed"
    assert data["stats"]["obligation_checks"] > 0
    assert data["stats"]["pruned_conflicts"] > 0
    assert data["provenance"]["independent_audits"] == ["independent two-pointer audit", "forward/reverse SHA-256"]
    assert all(row["audit"]["sha_equal"] == row["audit"]["exact"] for row in data["controls"])
    assert all(not row["provenance"]["reversed_finished_sentence"] for row in data["rendered_candidates"])
