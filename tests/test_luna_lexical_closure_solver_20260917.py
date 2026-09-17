import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "experiments/luna_lexical_closure_solver_20260917.py"
RUN = ROOT / "runs/luna-lexical-closure-solver-20260917.json"


def test_lexical_closure_contract():
    subprocess.run([sys.executable, str(SCRIPT)], check=True)
    data = json.loads(RUN.read_text())
    row = data["rows"][0]
    assert row["audit"]["letters"] > 38
    assert row["audit"]["exact"] is True
    assert row["audit"]["independent_two_pointer"]["exact"] is True
    assert row["audit"]["sha256_equal"] is True
    assert row["closure_search"]["memo_states"] > 0
    assert row["provenance"]["source_sentences_copied"] is False
    assert row["anti_shortcut"]["self_collision"] is True
    assert row["admitted"] is False
    assert row["next_repair"]
