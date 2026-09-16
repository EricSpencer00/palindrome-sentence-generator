import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_entailment_probe_has_independent_prose_and_audits():
    subprocess.run([sys.executable,"experiments/entailment_rewrite_lattice_20260916.py"],cwd=ROOT,check=True,capture_output=True,text=True)
    r=json.loads((ROOT/"runs/entailment-rewrite-lattice-20260916.json").read_text())
    assert not r["novelty_preflight"]["passed"]
    assert r["novelty_preflight"]["related_families"]
    assert r["candidate"]["left"]["text"].endswith(".") and r["candidate"]["right"]["text"].endswith(".")
    assert r["candidate"]["left"]["audit"]["sha256_forward"] != r["candidate"]["left"]["audit"]["sha256_reverse"]
    assert r["candidate"]["provenance"]["catalogue_text_used"] is False
