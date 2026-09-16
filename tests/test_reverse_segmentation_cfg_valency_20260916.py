import json
from pathlib import Path

from experiments.reverse_segmentation_cfg_valency_20260916 import run


def test_reverse_segmentation_has_complete_prose_and_independent_audits():
    out = run()
    assert out["novelty_preflight"]["passed"]
    assert len(out["rows"]) == 3
    for row in out["rows"]:
        assert row["rendered"].endswith(".")
        assert row["length"] >= 40
        assert row["exact_audit"]["exact"] is False
        assert row["exact_audit"]["two_pointer"] is False
        assert row["shortcut_gate"]["complete_clause"]
        assert row["provenance"]["generated_not_catalogue"]
        assert row["next_repair"]


def test_registry_does_not_claim_reader_eligibility():
    data = json.loads(Path("docs/experiment-novelty-registry.json").read_text())
    assert all(e.get("id") != "reverse-segmentation-cfg-valency-20260916" for e in data["entries"])
