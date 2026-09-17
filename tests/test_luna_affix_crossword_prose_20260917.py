import json
from pathlib import Path

from experiments import luna_affix_crossword_prose_20260917 as lane


def test_affix_crossword_run_has_complete_prose_and_independent_audits():
    result = lane.run()
    assert result["status"] == "completed_no_exact_closure"
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["longest_letters"] > 80
    assert result["stats"]["repaired_variants"] == 24
    assert result["stats"]["adjunct_object_variants"] == 4
    assert result["stats"]["subject_theme_variants"] == 4
    assert result["stats"]["verb_clitic_variants"] == 4
    assert result["stats"]["preposition_locative_variants"] == 4
    assert result["stats"]["determiner_adjective_variants"] == 4
    assert {row["repair"] for row in result["candidates"]} >= {
        "past-tense + object-clitic", "present-tense + speaker-clitic"
    }
    assert result["candidates"]
    for row in result["candidates"]:
        assert row["audit"]["independent_two_pointer_exact"] == row["audit"]["exact"]
        assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]
        assert row["plan"]["obligations_checked_before_render"] is True
        assert row["reader_eligible"] is False


def test_affix_lane_run_artifact_matches_generator():
    path = Path("runs/luna-affix-crossword-prose-20260917.json")
    lane.OUT.parent.mkdir(exist_ok=True)
    result = lane.run()
    path.write_text(json.dumps(result, indent=2) + "\n")
    saved = json.loads(path.read_text())
    assert saved["experiment_id"] == lane.EXPERIMENT_ID
    assert saved["actual_prose"]
    assert saved["provenance"]["catalogue_text_imported"] is False
