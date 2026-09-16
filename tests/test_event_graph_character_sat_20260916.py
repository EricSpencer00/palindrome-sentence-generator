import json
from pathlib import Path

ROOT = Path(__file__).parents[1]

def test_event_graph_run_has_dual_agreeing_audits_and_provenance():
    d = json.loads((ROOT / "runs/event-graph-character-sat-20260916.json").read_text())
    assert d["novelty_preflight"]["passed"]
    assert d["stats"]["probes"] == 81
    assert d["stats"]["exact"] == 0
    assert d["stats"]["dual_disagreements"] == 0
    assert d["provenance"]["reverse_emitter"] is False
    assert all(r["reader_eligible"] is False for r in d["probes"])
    assert all(r["exact_audit"]["exact"] == r["independent_audit"]["exact"] for r in d["probes"])

def test_longest_probe_is_intact_sentence_pair():
    d = json.loads((ROOT / "runs/event-graph-character-sat-20260916.json").read_text())
    row = max(d["probes"], key=lambda x: x["length"])
    assert row["length"] >= 60
    assert row["rendered"].count(".") == 2
    assert row["shortcut_checks"]["word_order_symmetry"] is False
