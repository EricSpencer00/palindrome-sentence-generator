import json, subprocess, sys
from pathlib import Path

def test_subject_continuity_run():
    subprocess.run([sys.executable, "experiments/subject_continuity_prelexical_csp_20260920.py"], check=True)
    d = json.loads(Path("runs/subject-continuity-prelexical-csp-20260920.json").read_text())
    assert d["counts"]["controls"] == 32
    assert d["counts"]["live_equations"] == 32
    assert d["counts"]["exact_over_38"] == 0
    assert all("pointer_audit" in c["audit"] and "sha256" in c["audit"] for c in d["controls"])
    assert all(c["audit"]["sha256"] == c["audit"]["reverse_sha256"] for c in d["controls"] if c["audit"]["exact"])
