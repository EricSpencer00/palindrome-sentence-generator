from experiments.pos_bilateral_cfg_orbit_20260920 import audit, search


def test_typed_pos_orbit_stops_without_relaxing_exactness():
    result = search(("det", "noun", "verb", "det", "noun"), max_nodes=2000, beam_width=200)
    assert result["nodes"] > 0
    assert result["status"] in {"exhausted", "node_budget"}
    assert result["closures"] == []


def test_independent_audit_is_literal_character_level():
    row = audit("ab ba")
    assert row["exact"] is True
    non = audit("a calm sailor reads a clear map")
    assert non["exact"] is False
    assert non["forward_sha256"] != non["reverse_sha256"]
