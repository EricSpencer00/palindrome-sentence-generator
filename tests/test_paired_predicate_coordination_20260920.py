import json
import subprocess
import sys
from pathlib import Path


def test_paired_predicate_coordination_run():
    subprocess.run([sys.executable, "experiments/paired_predicate_coordination_20260920.py"], check=True)
    data = json.loads(Path("runs/paired-predicate-coordination-20260920.json").read_text())
    assert data["counts"]["controls"] == 45
    assert data["counts"]["exact"] == 45
    assert data["counts"]["exact_over_38"] == 0
    assert all(c["audit"]["pointer_audit"] and c["audit"]["sha256"] == c["audit"]["reverse_sha256"] for c in data["controls"])
    assert all(c["novelty"]["novel"] for c in data["controls"])
