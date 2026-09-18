import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.two_sided_relative_balance_20260917 import run, letters

def test_balanced_two_sided_repair_audits_every_rendering():
    result = run()
    assert result["stats"]["rendered"] >= 1
    assert result["stats"]["exact"] == 0
    for row in result["rendered_candidates"]:
        assert row["length_balance"]["left_minus_right_delta"] == row["length_balance"]["reference_delta"]
        tape = letters(row["rendered"])
        assert row["audit"]["two_pointer_exact"] == (tape == tape[::-1])
        assert row["provenance"]["coordinated_two_sided"]
