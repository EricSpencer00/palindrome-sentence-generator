from boundary_conditioned_semantic_scene_search_20260921 import run


def test_terminal_index_selects_compatible_outer_pair_and_grows_inward():
    result = run()
    assert result["config"]["global_equation"] == "x[i] = x[N-1-i]"
    assert result["stats"]["boundary_conditioned_attempts"] > 0
    assert result["stats"]["max_committed_pairs"] >= 2
    for row in result["controls"]:
        assert row["provenance"]["anti_shortcut"]["finished_tape_reversal"] is False
        assert row["provenance"]["anti_shortcut"]["post_render_repair"] is False
        assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]
    assert result["novelty_preflight"]["status"] == "passed"
