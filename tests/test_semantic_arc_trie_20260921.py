import json, runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOD = runpy.run_path(str(ROOT / "experiments/semantic_arc_trie_20260921.py"))

def test_arc_run_has_rendered_provenance_and_independent_audit():
    MOD["main"]()
    data = json.loads((ROOT / "runs/semantic-arc-trie-20260921.json").read_text())
    assert data["candidate_count"] == 36
    assert data["exact_count"] == 0
    assert data["reader_eligible"] is False
    assert data["provenance"]["shortcuts_excluded"] is True
    for row in data["rendered_candidates"]:
        assert row["rendered"]
        assert row["audit"]["letters"] > 38
        assert row["audit"]["forward_sha256"] != row["audit"]["reverse_sha256"]
        assert row["live_trace"]["center_inside_word"] is True

def test_audit_rejects_a_near_miss():
    assert MOD["audit"]("A man, a plan.")["two_pointer_exact"] is False
