import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from experiments.paired_obligation_astar_20260916 import _audit, run


def test_astar_run_records_negative_result_without_readability_claim():
    result = run(state_limit=2_000)
    assert result["novelty_preflight"]["registry_entries_before_run"] == 97
    assert result["stats"]["states"] == 2_000
    assert result["stats"]["reader_eligible"] == 0
    assert result["provenance"]["programmatic_readability_claim"] is False
    assert result["reader_gate"]["status"] == "not_run"
    assert result["rendered_candidates_and_probes"]


def test_independent_audit_agrees_on_exactness():
    audit = _audit("Able was I ere I saw Elba.")
    assert audit["exact"] is True
    assert audit["independent_exact"] is True
    assert audit["two_pointer_exact"] is True
