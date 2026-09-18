import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.frame_pair_seam_csp_20260917 import run, letters

def test_csp_filters_before_render_and_audits_survivors():
    result = run()
    assert result["stats"]["rejected_pre_render"] > 0
    assert result["stats"]["rendered"] > 0
    assert result["stats"]["exact"] == 0
    for row in result["rendered_candidates"]:
        assert row["csp"]["satisfied"]
        tape = letters(row["rendered"])
        assert row["audit"]["two_pointer_exact"] == (tape == tape[::-1])
