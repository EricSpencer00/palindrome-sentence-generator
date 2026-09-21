from pathlib import Path
from experiments.executable_discourse_pronoun_accessibility_20260920 import audit, run, accessible_forms


def test_typed_binding_prunes_before_rendering():
    result = run()
    assert result["stats"]["semantic_prunes"] > 0
    assert result["stats"]["semantic_valid_pairs"] > 0
    assert result["stats"]["exact_gt38"] == 0
    assert result["semantic_rejection_witnesses"][0]["roles_unresolved"] is True
    assert result["provenance"]["typed_pronoun_accessibility"] is True
    assert Path("runs/executable-discourse-pronoun-accessibility-20260920.json").exists()


def test_pronouns_require_accessible_antecedent():
    event = {"subject": "courier", "recipient": "archivist", "theme": "map"}
    assert all(not row["pronouns"] for row in accessible_forms(event, set()))
    assert any(row["pronouns"] for row in accessible_forms(event, {"map"}))


def test_audit_rejects_control():
    assert not audit("The archivist sent the courier the map, then the courier studied the map.")["pointer_exact"]
