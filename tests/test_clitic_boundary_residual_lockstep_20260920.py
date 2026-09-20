import json, subprocess
from pathlib import Path
ROOT=Path(__file__).parents[1]
def test_clitic_residual_run():
    subprocess.run(["python3",str(ROOT/"experiments/clitic_boundary_residual_lockstep_20260920.py")],check=True)
    d=json.loads((ROOT/"runs/clitic-boundary-residual-lockstep-20260920.json").read_text())
    assert d["signature"]=="lockstep|clitic-boundary|agreement|full-residual-vector|independent-frames"
    assert d["stats"]["frame_count"]==18
    assert d["stats"]["fresh_exact_gt38"]==0
    assert all("residual_vector" in r and "audit" in r for r in d["rendered_candidates"])
