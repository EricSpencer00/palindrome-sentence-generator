from experiments.scene_event_lattice_20260930 import build_lattice, run


def test_anaphora_is_semantically_gated():
    g, trace = build_lattice()
    assert any(t["event"] == "diana_read_memos" and t["requires"] == ["Diana", "memos"]
               for t in trace)
    assert all(t["target_mask"] != 4 for t in trace)


def test_lattice_runs_with_audited_controls():
    result = run()
    assert result["provenance"]["complete_sentence_enumeration"] is False
    assert len(result["controls"]) == 2
    assert all("audit" in row for row in result["controls"])
    assert all(row["audit"]["exact"] is False for row in result["controls"])
