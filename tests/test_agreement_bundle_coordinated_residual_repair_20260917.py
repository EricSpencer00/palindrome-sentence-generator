import json
from pathlib import Path

RUN = Path(__file__).parents[1] / "runs/agreement-bundle-coordinated-residual-repair-20260917.json"

def test_all_rows_have_independent_exact_audit_and_prose():
    data = json.loads(RUN.read_text())
    assert data["candidate_count"] == len(data["candidates"]) > 0
    for row in data["candidates"]:
        assert row["audit"]["exact"] == row["audit"]["independent_two_pointer"]
        assert len(row["audit"]["sha256"]) == 64
        assert row["anti_shortcut"]["intact_prose"]
        assert row["provenance"]

def test_bundles_are_agreement_carrying():
    data = json.loads(RUN.read_text())
    valid = {("gardener", "carries", "letters"), ("teacher", "writes", "notes"),
             ("cartographer", "marks", "maps"), ("messenger", "records", "charts"),
             ("archivist", "keeps", "records")}
    assert all(tuple(r["bundle"]) in valid for r in data["candidates"])
