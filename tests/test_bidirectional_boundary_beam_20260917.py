import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/bidirectional-boundary-beam-20260917.json"


def test_boundary_beam_emits_independently_audited_resegmented_candidates():
    subprocess.run([sys.executable, "experiments/bidirectional_boundary_beam.py"], cwd=ROOT, check=True)
    payload = json.loads(RUN.read_text())
    assert payload["candidates"]
    assert payload["novelty_preflight"]["signature_collision"] is False
    assert payload["next_repair"]["operator"] == "tagged_pos_trie_and_phrase_score"
    for row in payload["candidates"]:
        assert row["exact"] is True
        assert row["reverse_tape"] is True
        assert row["sha256_forward"] == row["sha256_reverse"]
        assert row["source_kind"] in {"authored_seed", "labelled_fixture"}
        assert "|" in row["rendered"]
