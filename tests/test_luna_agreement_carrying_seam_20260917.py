import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import luna_agreement_carrying_seam_20260917 as lane

def test_morphology_first_lane_audits_every_rendered_candidate():
    result = lane.run()
    assert result["status"] == "completed_no_exact_closure"
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["candidate_count"] == 10
    assert result["stats"]["morphology_states"] == 4
    for row in result["candidates"]:
        assert row["morphology_before_lexicalization"]
        assert row["semantic_valency"]["left"]["frame"] == "transitive"
        assert row["seam"]["checked_before_render"]
        assert row["audit"]["independent_two_pointer_exact"] == row["audit"]["exact"]
        assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]
        assert row["anti_shortcut"]["finished_sentence_reversal"] is False
    assert any(r["repair"] == "held-out agreement carry repair" for r in result["candidates"])

def test_artifact_contains_all_rendered_candidates():
    lane.run()
    lane.OUT.parent.mkdir(exist_ok=True)
    lane.OUT.write_text(json.dumps(lane.run(), indent=2) + "\n")
    saved = json.loads(lane.OUT.read_text())
    assert len(saved["actual_prose"]) == saved["candidate_count"]
