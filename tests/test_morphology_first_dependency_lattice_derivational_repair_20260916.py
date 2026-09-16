from experiments.morphology_first_dependency_lattice_derivational_repair_20260916 import (
    STATE_SPACE_SIGNATURE,
    novelty_preflight,
    run,
)


def test_repair_preflight_accepts_only_registered_same_family():
    preflight = novelty_preflight()
    assert preflight["status"] == "same_family_repair"
    assert preflight["same_family_base"] is True
    assert preflight["foreign_signature_collision"] == []
    assert preflight["foreign_artifact_collision"] == []
    assert "lemma-derivation-inflection-path" in STATE_SPACE_SIGNATURE


def test_repair_preserves_negative_evidence_without_readability_claim():
    result = run()
    assert result["stats"]["tree_count"] == 6
    assert result["stats"]["raw_yields"] == 24570
    assert result["stats"]["exact_yields"] == 0
    assert result["stats"]["mechanically_admitted"] == 0
    assert result["negative_evidence_near_misses"]
    assert result["reader_gate"]["status"] == "not_run"
    assert result["provenance"]["readability_certificate"] is False
