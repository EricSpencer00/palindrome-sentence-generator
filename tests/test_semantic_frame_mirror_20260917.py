import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.semantic_frame_mirror_20260917 import run, letters


def test_bounded_semantic_frame_outputs_have_independent_audits():
    result = run()
    assert result["stats"]["rendered"] == 4
    assert result["stats"]["exact"] == 0
    for row in result["rendered_candidates"]:
        tape = letters(row["rendered"])
        assert row["audit"]["sha256_forward"]
        assert row["audit"]["sha256_reverse"]
        assert row["audit"]["two_pointer_exact"] == (tape == tape[::-1])
        assert row["provenance"]["semantic_frame_authored"]
        assert not row["provenance"]["catalogue_used"]
