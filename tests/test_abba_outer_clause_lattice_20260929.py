import hashlib
import json
import re
from pathlib import Path

from experiments.abba_outer_clause_lattice_20260929 import run


def test_live_lattice_and_independent_audit():
    result = run()
    assert result["stats"]["live_seam_attempts"] == 49
    assert result["stats"]["closed_outer_pairs"] == 1
    row = result["rendered_candidates"][0]
    tape = re.sub(r"[^a-z]", "", row["rendered"].lower())
    assert row["audit"]["two_pointer_exact"]
    assert tape == tape[::-1]
    assert row["audit"]["sha256_forward"] == hashlib.sha256(tape.encode()).hexdigest()
    assert row["audit"]["sha256_reverse"] == hashlib.sha256(tape[::-1].encode()).hexdigest()
    assert row["provenance"]["live_reverse_obligation"]


def test_artifact_is_reproducible_json():
    result = run()
    saved = json.loads(Path("runs/abba-outer-clause-lattice-20260929.json").read_text())
    assert saved["stats"] == result["stats"]
    assert saved["rendered_candidates"] == result["rendered_candidates"]
