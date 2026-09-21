import json
from pathlib import Path

RUN = Path(__file__).parents[1] / "runs" / "corpus-role-online-csp-20260921.json"

def test_role_bank_run_is_reproducible_and_long_shaped():
    x=json.loads(RUN.read_text())
    assert x["method"].startswith("corpus role bank")
    assert x["source_count"] >= 43
    assert x["total_nodes"] > 0
    assert max((c["letters"] for c in x["controls"]), default=0) > 38

def test_any_candidate_has_independent_provenance_and_exact_gate():
    x=json.loads(RUN.read_text())
    for c in x["candidates"]:
        assert c["exact"] is True
        assert c["novel_ngrams"] is True
        assert c["source_id"] is not None and c["sha"]
