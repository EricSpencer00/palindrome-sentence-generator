import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.flexible_frame_slot_automata_20260917 import run

def test_flexible_lane_contract_and_audits():
    out = run()
    assert out["status"] == "completed_no_exact_closure"
    assert set(out["stats"]["frames"]) == {"declarative", "question", "imperative"}
    assert out["stats"]["longest_letters"] > 38
    assert out["novelty_preflight"]["exact_signature_collision"] is False
    assert out["rows"]
    for row in out["rows"]:
        assert row["boundary_resegmentation"]
        assert row["provenance"]["agreement_checked_before_admission"]
        assert row["exact_audit"]["independent_agreement"]
