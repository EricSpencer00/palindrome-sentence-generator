import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from experiments.dream_rsi_exact_boundary_20260918 import audit, run


def test_exact_boundary_lane_keeps_fresh_and_withheld_evidence_separate():
    payload = run()
    assert payload["method"].startswith("Dream-RSI exact-boundary")
    assert payload["stats"]["fresh_exact"] == 0
    assert payload["stats"]["withheld_smoke_exact"] > 0
    assert payload["reader_gate"]["status"] == "not_triggered"
    # Independent pointer and digest checks must agree on every rendered row.
    for row in payload["rendered_candidates"]:
        checked = audit(row["rendered"])
        assert checked == row["audit"]
        assert not checked["two_pointer_exact"]
        assert row["provenance"]["fresh_authored_control"]
    # The smoke closure is mechanically real but cannot be promoted into a
    # fresh result or reader package.
    texts = {row["rendered"] for row in payload["withheld_smoke_closures"]}
    assert "an aide rips nine memos; some men inspire Diana." in texts
    assert all(row["audit"]["two_pointer_exact"] for row in payload["withheld_smoke_closures"])
    assert payload["provenance"]["withheld_smoke_fixture_is_not_a_candidate"]


def test_run_artifact_matches_recomputed_payload():
    payload = run()
    path = Path("runs/dream-rsi-exact-boundary-20260918.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    stored = json.loads(path.read_text())
    assert stored["stats"] == payload["stats"]
