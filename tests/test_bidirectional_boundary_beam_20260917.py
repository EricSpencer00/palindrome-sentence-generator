import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/bidirectional-boundary-beam-20260917.json"


def test_boundary_beam_records_typed_empty_frontier_and_concrete_repair():
    subprocess.run([sys.executable, "experiments/bidirectional_boundary_beam.py"], cwd=ROOT, check=True)
    payload = json.loads(RUN.read_text())
    # The tightened POS lattice is intentionally fail-closed: it may return
    # no complete clause rather than relabeling a fragment as prose.
    assert payload["candidates"] == []
    assert payload["novelty_preflight"]["signature_collision"] is False
    assert payload["next_repair"]["operator"] == "lexical_expansion"
    assert payload["next_repair"]["target"] == "add attested verb senses without lowering the frequency floor"
