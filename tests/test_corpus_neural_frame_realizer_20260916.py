import json
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/corpus-neural-frame-realizer-20260916.json"
spec=importlib.util.spec_from_file_location("realizer", ROOT/"experiments/corpus_neural_frame_realizer_20260916.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
EXPERIMENT_ID, SIGNATURE, run, exact = mod.EXPERIMENT_ID, mod.SIGNATURE, mod.run, mod.exact

def test_frozen_run_has_hard_gates_and_no_readability_claim():
    payload=json.loads(OUT.read_text())
    assert payload == run()
    assert payload["signature"] == SIGNATURE
    assert payload["stats"]["admitted"] == 0
    assert payload["stats"]["reader_eligible"] == 0
    assert "No readability claim" in payload["readability_note"]
    assert all(exact(r["text"]) for r in payload["exact_audit"])

def test_registry_entry_is_unique_and_artifacts_exist():
    root=ROOT
    reg=json.loads((root/"docs/experiment-novelty-registry.json").read_text())
    rows=[x for x in reg["entries"] if x["id"]==EXPERIMENT_ID]
    assert len(rows)==1 and rows[0]["signature"]==SIGNATURE
    assert (root/rows[0]["artifact"]).exists()
    assert all((root/x).exists() for x in rows[0]["run_artifacts"])
