import json
from pathlib import Path

from experiments.luna_punctuation_center_bridge_20260917 import run, OUT


def test_punctuation_center_bridge_has_prose_and_independent_audits():
    result = run()
    assert result["novelty_preflight"]["passed"]
    assert result["candidate_count"] == 48
    assert result["exact_count"] == 0
    for row in result["rendered_candidates"]:
        assert row["complete_grammar"]
        assert row["independent_exact_agreement"]
        assert row["anti_shortcut_flags"]["punctuation_changes_letters"] is False
        assert row["center_event"]["text"] in row["rendered"]
        assert row["residual_repair"] is not None
    repaired = [r for r in result["rendered_candidates"] if r["repair_stage"] != "baseline"]
    assert len(repaired) == 36
    subject_repaired = [r for r in repaired if r["repair_stage"] == "held_out_center_adjacent_subject"]
    object_repaired = [r for r in repaired if r["repair_stage"] == "held_out_right_object_np"]
    locative_repaired = [r for r in repaired if r["repair_stage"] == "held_out_right_locative"]
    assert len(subject_repaired) == 12
    assert len(object_repaired) == 12
    assert len(locative_repaired) == 12
    assert all(r["held_out_slot"] == "right_subject_np" for r in subject_repaired)
    assert all(r["held_out_slot"] == "right_object_np" for r in object_repaired)
    assert all(r["held_out_slot"] == "right_locative" for r in locative_repaired)
    assert result["repair_summary"]["preserved_center_event"]
    assert result["repair_summary"]["preserved_punctuation"]
    assert result["repair_summary"]["held_out_slots"] == ["right_subject_np", "right_object_np", "right_locative"]


def test_run_artifact_is_reproducible_after_write():
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    saved = json.loads(OUT.read_text())
    assert saved["experiment"] == result["experiment"]
    assert saved["rendered_candidates"][0]["rendered"] == result["rendered_candidates"][0]["rendered"]
