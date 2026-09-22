import json
from pathlib import Path
from experiments.latent_discourse_storyboard_20260920 import run

def test_storyboard_has_novelty_and_controls():
    result = run()
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["scene_states"] == 12
    assert result["stats"]["rendered_controls"] == result["stats"]["rendered_candidates"]
    assert result["stats"]["live_frontier_prunes"] > 0
    assert result["controls"]
    assert all(not row["reader_eligible"] for row in result["candidates"])

def test_audits_are_independent_and_provenance_is_clean():
    result = run()
    for row in result["candidates"]:
        assert row["audit"]["sha256_forward"] != ""
        assert row["provenance"]["online_character_frontier"]
        assert not row["provenance"]["finished_tape_reversal"]
        assert not row["provenance"]["posthoc_repair"]
