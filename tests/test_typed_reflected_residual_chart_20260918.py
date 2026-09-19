import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUN = ROOT / "runs" / "typed-reflected-residual-chart-20260918.json"


def test_chart_run_records_live_residual_and_independent_gate():
    payload = json.loads(RUN.read_text())
    assert payload["construction"]["live_residual"] is True
    assert payload["construction"]["finished_tape_reversal"] is False
    assert payload["stats"]["exact_closures"] == 0
    assert payload["reader_gate"]["status"] == "closed"
    for row in payload["rendered_controls"]:
        assert row["audit"]["two_pointer_exact"] == row["audit"]["exact"]
        assert row["audit"]["sha_equal"] == row["audit"]["exact"]


def test_withheld_seed_is_not_promoted():
    payload = json.loads(RUN.read_text())
    smoke = payload["withheld_seed_smoke"]
    assert smoke["used_as_output"] is False
    assert smoke["audit"]["exact"] is True
