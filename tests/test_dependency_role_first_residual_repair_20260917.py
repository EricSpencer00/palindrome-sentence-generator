import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "dependency-role-first-residual-repair-20260917.json"


def test_targeted_role_repair_is_fresh_and_bounded():
    data = json.loads(RUN.read_text())
    assert data["experiment"] == "dependency-role-first-residual-repair-20260917"
    assert data["novelty_preflight"]["passed"] is True
    assert data["states_examined"] == 5
    assert data["exact_count"] == 0
    assert data["mechanically_admitted_count"] == 0
    assert len(data["best_rendered_candidates"]) == 5


def test_each_candidate_has_intact_prose_independent_checks_and_provenance():
    data = json.loads(RUN.read_text())
    for candidate in data["best_rendered_candidates"]:
        assert candidate["letters"] > 100
        assert candidate["rendered"].endswith(".")
        assert candidate["exact_check_two_pointer"]["algorithm"] == "independent_two_pointer"
        assert candidate["exact_check_sha256"]["algorithm"] == "independent_forward_reverse_sha256"
        assert candidate["independent_exact_agreement"] is True
        assert candidate["replacement"]["position"] == 1
        assert candidate["provenance"]["source_sentences_copied"] is False
        assert candidate["provenance"]["catalogue_imported"] is False
        assert candidate["anti_shortcut_flags"]["isolated_character_edit"] is False
        assert candidate["anti_shortcut_flags"]["complete_constituents_only"] is True
        assert candidate["next_repair"]

