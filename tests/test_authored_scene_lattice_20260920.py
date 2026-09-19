from experiments.authored_scene_lattice_20260920 import run

def test_lattice_has_independently_exact_rows_and_no_hidden_reversal():
    result = run()
    assert result["stats"]["nodes"] == 45
    assert result["stats"]["exact"] == 45
    assert result["stats"]["longest_exact_letters"] == 42
    assert all(row["audit"]["two_pointer_exact"] for row in result["candidates"])
    assert all(not row["provenance"]["finished_tape_reversed"] for row in result["candidates"])
