import json
from pathlib import Path

RUN = Path(__file__).parents[1] / "runs/semantic-live-residual-role-repair-20260917.json"

def test_run_has_independent_audits_and_provenance():
    data = json.loads(RUN.read_text())
    assert data["candidate_count"] == len(data["candidates"]) > 0
    for row in data["candidates"]:
        a = row["audit"]
        assert a["independent_two_pointer"] == a["exact"]
        assert a["sha256"]
        assert row["provenance"]
        assert row["anti_shortcut"]["intact_prose"]

def test_only_same_role_repairs_are_recorded():
    data = json.loads(RUN.read_text())
    for row in data["candidates"]:
        assert row["repair_role"] is None or row["repair_role"] in {"agent", "verb", "object", "setting"}
