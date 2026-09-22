import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_typed_abba_run_has_independent_exact_closures():
    subprocess.run([sys.executable, "experiments/grammar_abba_paragraph_20260928.py"],
                   cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH":"."}, check=True)
    payload = json.loads((ROOT / "runs/grammar-abba-paragraph-20260928.json").read_text())
    assert payload["stats"] == {"clause_frames": 3, "pairs": 3, "exact": 3, "longest_letters": 46}
    best = max(payload["exact_candidates"], key=lambda x: x["letters"])
    assert best["rendered"] == "Nora, I saw evil. Noel, I saw war. Raw was I, Leon. Live was I, Aron."
    assert best["audit"]["two_pointer_exact"]
    assert best["audit"]["sha_equal"]
    assert best["audit"]["validator_exact"]
    assert best["novelty_preflight"]
    assert best["provenance"]["per_candidate_rlaif"] is False
    assert best["provenance"]["finished_tape_reversal"] is False
