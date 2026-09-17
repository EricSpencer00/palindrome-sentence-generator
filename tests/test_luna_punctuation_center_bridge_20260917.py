import json
from pathlib import Path

from experiments.luna_punctuation_center_bridge_20260917 import run, OUT


def test_punctuation_center_bridge_has_prose_and_independent_audits():
    result = run()
    assert result["novelty_preflight"]["passed"]
    assert result["candidate_count"] == 12
    assert result["exact_count"] == 0
    for row in result["rendered_candidates"]:
        assert row["complete_grammar"]
        assert row["independent_exact_agreement"]
        assert row["anti_shortcut_flags"]["punctuation_changes_letters"] is False
        assert row["center_event"]["text"] in row["rendered"]
        assert row["residual_repair"] is not None


def test_run_artifact_is_reproducible_after_write():
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    saved = json.loads(OUT.read_text())
    assert saved["experiment"] == result["experiment"]
    assert saved["rendered_candidates"][0]["rendered"] == result["rendered_candidates"][0]["rendered"]
