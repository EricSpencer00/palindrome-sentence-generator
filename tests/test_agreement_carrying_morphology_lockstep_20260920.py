import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).parents[1]
RUN = ROOT / "runs" / "agreement-carrying-morphology-lockstep-20260920.json"

def test_lockstep_run_is_reproducible_and_audited():
    subprocess.run(["python3", str(ROOT / "experiments/agreement_carrying_morphology_lockstep_20260920.py")], check=True)
    data = json.loads(RUN.read_text())
    assert data["signature"] == "lockstep|agreement-inflection|auxiliary-clitic-boundaries|live-obligation"
    assert data["stats"]["transitions"] == 2916
    assert data["stats"]["fresh_exact_gt38"] == 0
    assert len(data["rendered_candidates"]) == 2
    for row in data["rendered_candidates"]:
        assert row["audit"]["two_pointer_exact"] is False
        assert row["provenance"]["finished_tape_reversal"] is False
