from experiments.phrase_path_graph_join_20260920 import PATTERNS, run


def test_phrase_patterns_have_transitive_shapes():
    assert ("DET", "N", "V", "NUM", "N") in PATTERNS


def test_phrase_graph_join_is_bounded():
    result = run(pos_limit=15, edge_limit=30000, fanout=20, max_left_paths=10000)
    assert result["stats"]["status"] in {"SAT", "UNSAT"}
    assert result["provenance"]["typed_right_parser"]
