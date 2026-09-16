import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run():
    p = subprocess.run([sys.executable, str(ROOT / "experiments/bilateral_semantic_cfg_20260916.py")],
                       check=True, capture_output=True, text=True)
    return json.loads(p.stdout)

def test_bilateral_cfg_is_complete_prose_and_independently_audited():
    d = run(); c = d["candidate"]
    assert c["audits"]["letters"] >= 60
    assert not c["audits"]["exact"]
    assert c["audits"]["two_pointer"] is False
    assert c["audits"]["hash_equal"] is False
    assert len(d["frontier_trace"]) == 3
    assert d["construction"]["fixed_tape"] is False
    assert d["construction"]["ordinary_word_order"] is True
    assert d["repair"]["after"] != c["text"]

def test_repair_is_held_out_semantic_recomputation():
    d = run()
    assert d["repair"]["operator"] == "held-out semantic object substitution"
    assert "the lock" in d["repair"]["after"]
    assert d["provenance"]["generator_sha256"]
