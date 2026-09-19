from experiments.compositional_shell_seam_dp_20260920 import run


def test_compositional_shells_keep_readable_prose_live_but_do_not_claim_exactness():
    result = run()
    assert result["stats"] == {
        "visited": 21660,
        "retained": 164,
        "exact": 0,
        "longest_retained_letters": 75,
        "best_seam_match_chars": 4,
    }
    assert result["candidates"][0]["audit"]["two_pointer_exact"] is False
    assert result["candidates"][0]["provenance"]["finished_tape_reversed"] is False
    assert result["reader_gate"] == "closed until blinded readers judge intact prose"
