"""Regression checks for the clean repeated-shell event lattice."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "incumbent-568-repeated-shell-event-lattice-20261002.json"


def test_authoritative_lineage_and_rejected_evidence() -> None:
    payload = json.loads(ARTIFACT.read_text())
    assert payload["parent"] == {
        "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
        "id": "outer-causal-scene-568-working-incumbent",
        "letters": 568,
        "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
    }
    assert [entry["letters"] for entry in payload["preserved_frontier"]] == [568, 560, 558, 556]
    assert {entry["letters"] for entry in payload["rejected_evidence"] if entry["status"] != "exact_prior_evidence"} == {592, 596}
    assert payload["comparison_evidence"]["source_commit"] == "9cb68296"
    assert payload["config"]["historical_608_650_loaded"] is False


def test_shell_reflection_lattice_and_full_render_gate() -> None:
    row = json.loads(ARTIFACT.read_text())["rows"][0]
    assert row["seam"] == {
        "normalized_windows": [[64, 91], [477, 504]],
        "raw_spans": [[83, 120], [662, 700]],
        "old_left": "Mara stops rats. A tub? He maps Aron.",
        "old_right": "Nora, spam. Eh, but a star spots Aram.",
        "reflection_verified": True,
    }
    assert row["independent_audit"]["normalized_letters"] == 594
    assert row["independent_audit"]["two_pointer_exact"] is True
    assert row["independent_audit"]["sha_equal"] is True
    assert row["audit"]["project_validator_exact"] is True
    assert row["lattice"]["role_families"] == ["control", "observation"]
    assert row["online_state"]["final_residual"] == ""
    assert row["online_state"]["attempted_paired_expansions"] == 3
    assert all(attempt["status"] == "accepted" for attempt in row["online_state"]["attempts"])
    assert all(attempt["residual"]["final_residual"] == "" for attempt in row["online_state"]["attempts"])
    assert all(row["full_render_gate"][key] for key in row["full_render_gate"] if key != "status")
    assert row["malformed_exact_rejection_policy"]["blocked_phrases"] == ["Draw no maps", "Spam onward"]
