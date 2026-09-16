import json
from pathlib import Path

ROOT = Path(__file__).parents[1]

def test_run_has_independent_exact_audit_and_no_reader_shortcut():
    run = json.loads((ROOT / "runs/reversible-grammar-insertion-20260916.json").read_text())
    assert run["novelty_preflight"]["exact_signature_collisions"] == []
    assert run["novelty_preflight"]["exact_artifact_collisions"] == []
    assert run["exact_count"] == 5
    assert run["reader_eligible_count"] == 0
    for row in run["candidates"]:
        assert row["exact"] is True
        assert row["pair_reverse_check"] is True
        assert row["no_repeated_units"] is True
        assert row["reader_eligible"] is False
