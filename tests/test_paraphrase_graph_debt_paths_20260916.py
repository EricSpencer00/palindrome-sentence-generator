import json
from pathlib import Path
from experiments.paraphrase_graph_debt_paths_20260916 import EVIDENCE, run, exact, two_pointer, hash_check

def test_graph_evidence_has_long_independent_probes():
    p = run(); assert p["novelty_preflight"]["exact_signature_collision"] is False
    assert p["stats"]["probes"] >= 39
    assert p["stats"]["min_letters"] >= 39
    assert p["stats"]["exact"] == 0 and p["stats"]["admitted"] == 0

def test_four_checks_agree_on_packaged_probes_and_repair():
    p = run()
    for row in p["probes"]:
        vals = row["checks"]; assert vals["exact"] == vals["two_pointer"] == vals["hash"]
    vals = p["repair"]["checks"]; assert vals["exact"] == vals["two_pointer"] == vals["hash"]
    assert not exact("A readable sentence.") and not two_pointer("A readable sentence.") and not hash_check("A readable sentence.")

def test_registry_and_frozen_run_exist():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "docs/experiment-novelty-registry.json").read_text())
    assert any(e["id"] == "paraphrase-graph-debt-paths-20260916" for e in registry["entries"])
    assert EVIDENCE.exists()
