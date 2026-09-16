import json
from pathlib import Path

from experiments.centerout_grammar_boundary_repair_20260916 import run, EXPERIMENT_ID

ROOT = Path(__file__).resolve().parents[1]

def test_single_boundary_repair_is_targeted_and_audited():
    payload = run(); row = payload["candidate"]
    assert payload["stats"] == {"candidates": 1, "exact": 0, "mechanically_admitted": 0}
    assert row["repair"]["changed_component"] == "right adjunct only"
    assert row["provenance"]["authored_event_preserved"]
    assert row["exact_audit"]["two_pointer_exact"] is False
    assert row["exact_audit"]["sha_equal"] is False
    assert row["letters"] >= 39
    registry = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    assert any(x["id"] == EXPERIMENT_ID for x in registry["entries"])
