from pathlib import Path
from experiments.executable_discourse_recipient_theme_20260920 import audit, run


def test_recipient_theme_state_prunes_before_rendering():
    result = run()
    assert result["stats"]["semantic_prunes"] > 0
    assert result["stats"]["semantic_valid_pairs"] > 0
    assert result["stats"]["exact_gt38"] == 0
    assert result["semantic_rejection_witnesses"][0]["roles_unresolved"] is True
    assert Path("runs/executable-discourse-recipient-theme-20260920.json").exists()


def test_audit_rejects_control():
    assert not audit("The archivist sent the courier the map, then the courier studied the map.")["pointer_exact"]
