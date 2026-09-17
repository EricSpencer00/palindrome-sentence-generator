import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("center_splice", Path(__file__).parents[1] / "experiments" / "center_splice_clause_search_20260917.py")
LANE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LANE)


def test_center_splice_is_exact_but_shortcut_is_not_admitted():
    result = LANE.run()
    assert result["bounds"]["combinations"] == 64
    assert result["exact_candidates"]
    assert not result["admitted"]
    assert all(40 <= row["letters"] <= 100 for row in result["rejected_shortcut_candidates"])


def test_exactness_and_shortcut_control():
    text = "Deliver stressed drawer; level; reward desserts, reviled."
    assert LANE.letters(text) == LANE.letters(text)[::-1]
    assert LANE.semordnilap_shortcut("Deliver stressed drawer", "reward desserts reviled")
