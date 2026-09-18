import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.semantic_frame_seam_replacement_20260917 import run, letters

def test_authored_frame_replacement_is_independently_verified():
    result = run()
    assert result["stats"]["rendered"] == 4
    assert result["stats"]["exact"] == 0
    for row in result["rendered_candidates"]:
        tape = letters(row["rendered"])
        assert row["audit"]["two_pointer_exact"] == (tape == tape[::-1])
        assert row["provenance"]["authored_frame_replacement"]
        assert not row["provenance"]["catalogue_used"]
