from experiments.shared_scene_synchronous_transducer_20260921 import run


def test_shared_scene_transducer_has_novel_controls_and_no_false_exact_rows():
    result = run()
    assert result["stats"]["graph_states"] == 15
    assert result["stats"]["rendered"] == 15
    assert result["novelty_preflight"]["output_hashes_unique"] is True
    assert result["stats"]["exact_gt38"] == 0
    assert result["reader_facing_candidates"] == []
    assert all(row["audit"]["pointer_exact"] is False for row in result["diagnostic_controls"])
    assert all(row["audit"]["sha_equal"] is False for row in result["diagnostic_controls"])
