import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUN = ROOT / "runs" / "luna-recursive-balanced-grammar-20260917.json"


def test_recursive_balanced_grammar_emits_complete_prose_and_independent_audits():
    subprocess.run([sys.executable, str(ROOT / "experiments/luna_recursive_balanced_grammar_20260917.py")], check=True)
    data = json.loads(RUN.read_text())
    assert data["novelty_preflight"]["status"] == "passed"
    assert data["family"]["arbitrary_size"]
    assert len(data["rendered_candidates"]) == 5
    assert max(row["depth"] for row in data["rendered_candidates"]) == 2
    assert all(row["complete_prose"] for row in data["rendered_candidates"])
    assert all(row["pointer_audit"]["algorithm"] == "independent_two_pointer" for row in data["rendered_candidates"])
    assert all(row["hash_audit"]["algorithm"] == "independent_forward_reverse_sha256" for row in data["rendered_candidates"])
    assert max(row["letters"] for row in data["rendered_candidates"]) > 80
    assert data["next_repair"]["operator"]
