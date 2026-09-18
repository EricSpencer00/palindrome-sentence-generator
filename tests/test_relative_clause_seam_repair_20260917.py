import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.relative_clause_seam_repair_20260917 import run, letters

def test_seam_repair_is_bounded_and_independently_audited():
    result = run()
    assert result["stats"]["rendered"] == 8
    assert result["stats"]["exact"] == 0
    for row in result["rendered_candidates"]:
        tape = letters(row["rendered"])
        assert row["audit"]["two_pointer_exact"] == (tape == tape[::-1])
        assert row["provenance"]["seam_selected_before_repair"]
        assert not row["provenance"]["catalogue_used"]
