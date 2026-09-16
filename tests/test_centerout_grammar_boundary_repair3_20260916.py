import json
from pathlib import Path
from experiments.centerout_grammar_boundary_repair3_20260916 import run, EXPERIMENT_ID

ROOT=Path(__file__).resolve().parents[1]

def test_two_boundary_time_adjunct_repair():
    p=run(); r=p["candidate"]
    assert p["stats"] == {"candidates":1,"exact":0,"mechanically_admitted":0}
    assert r["repair"]["new_suffix"] == "ht"
    assert r["provenance"]["authored_event_preserved"]
    assert r["exact_audit"]["two_pointer_exact"] is False
    assert r["exact_audit"]["sha_equal"] is False
    assert r["rendered"].endswith("night.")
    assert len(r["rendered"].split(".")) == 3
    reg=json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text())
    assert any(x["id"]==EXPERIMENT_ID for x in reg["entries"])
