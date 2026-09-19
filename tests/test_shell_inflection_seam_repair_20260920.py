from experiments.shell_inflection_seam_repair_20260920 import run


def test_shell_inflection_repair_preserves_complete_svo_roles_without_claiming_closure():
    result = run()
    assert result["stats"] == {
        "visited": 11340,
        "retained": 420,
        "exact": 0,
        "longest_retained_letters": 27,
        "best_seam_match_chars": 4,
    }
    assert result["candidates"][0]["audit"]["two_pointer_exact"] is False
    assert result["candidates"][0]["provenance"]["finished_tape_reversed"] is False
    assert result["reader_gate"] == "closed until blinded readers judge intact prose"
