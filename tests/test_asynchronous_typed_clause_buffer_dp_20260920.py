import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "asynchronous-typed-clause-buffer-dp-20260920.json"


def test_async_buffer_run_is_reproducible_and_gated():
    subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "asynchronous_typed_clause_buffer_dp_20260920.py")],
        check=True,
        cwd=ROOT,
    )
    payload = json.loads(RUN.read_text())
    assert payload["novelty_preflight"]["status"] == "passed"
    assert payload["novelty_preflight"]["finished_tape_reversal"] is False
    assert payload["provenance"]["next_reader_test"]
    assert payload["stats"]["indexed_pairs"] == 0
    assert payload["stats"]["exact_gt38"] == 0
    assert "zero indexed pairs" in payload["next_construction"]
    for row in payload["exact_candidates"]:
        assert row["audit"]["exact"]
        assert row["audit"]["sha256_forward"] == row["audit"]["sha256_reverse"]
        assert row["provenance"]["reader_eligible"] is False
