"""Regression checks for the constructive masked-character lane."""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_masked_run_explores_real_complete_alternatives_and_stays_unadmitted():
    payload = json.loads((ROOT / "runs/masked-character-scene-gibbs-20260916.json").read_text())
    assert payload["novelty_preflight"]["passed"] is True
    assert len(payload["probes"]) == 3
    for probe in payload["probes"]:
        alternatives = probe["search"]["complete_assignments"]
        assert len(alternatives) == 3
        assert len({row["text"] for row in alternatives}) == 3
        assert probe["audit"]["exact"] is False
        assert probe["audit"]["two_pointer"] is False
        assert probe["audit"]["hash_equal"] is False
        assert probe["checks"]["length_band"] is True
    assert payload["reader_eligible"] is False


def test_masked_probe_length_matches_independent_tape():
    payload = json.loads((ROOT / "runs/masked-character-scene-gibbs-20260916.json").read_text())
    for probe in payload["probes"]:
        letters = len(re.findall(r"[A-Za-z]", probe["text"]))
        assert letters == probe["audit"]["tape_length"]
