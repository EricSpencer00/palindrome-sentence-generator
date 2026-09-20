import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).parents[1]

def test_compositional_artifact_and_controls():
    subprocess.run([sys.executable, str(ROOT / "experiments/compositional_non_nested_20260920.py")], check=True, cwd=ROOT)
    x = json.loads((ROOT / "runs/compositional-non-nested-20260920.json").read_text())
    assert x["stats"]["states"] > 0
    assert x["stats"]["fresh_exact_gt38"] == 0
    assert x["controls"][0]["audit"]["sha_equal"] is True
    assert all(c["audit"]["sha_equal"] is False for c in x["controls"][1:])
    assert x["novelty_preflight"]["finished_tape_reversal"] is False
    assert x["near_misses"]
