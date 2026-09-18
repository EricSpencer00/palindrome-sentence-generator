import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.relative_head_verb_window_20260917 import run, letters

def test_relaxed_window_is_bounded_and_audited():
    result = run()
    assert result["stats"]["rendered"] > 1
    assert result["stats"]["exact"] == 0
    for row in result["rendered_candidates"]:
        assert row["length_window"]["within_plus_minus_two"]
        tape = letters(row["rendered"])
        assert row["audit"]["two_pointer_exact"] == (tape == tape[::-1])
        assert row["provenance"]["semantic_head_and_verb_changed"]
