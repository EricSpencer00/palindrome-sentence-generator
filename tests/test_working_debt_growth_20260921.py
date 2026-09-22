import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "experiments" / "working_debt_growth_20260921.py"
spec = importlib.util.spec_from_file_location("working_debt", PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_seed_and_working_drafts_are_independently_exact():
    result = mod.run()
    assert result["seed_regression"]["audit"]["two_pointer_exact"]
    assert result["stats"]["exact_working_candidates"] == 3
    assert result["stats"]["longest_exact_working_letters"] == 106
    assert result["stats"]["growth_over_seed"] == 68
    for row in result["working_candidates"]:
        assert row["audit"]["two_pointer_exact"]
        assert row["audit"]["sha_equal"]
        assert row["exact_closure"]
        assert not row["provenance"]["reader_certified"]


def test_longest_draft_is_not_mislabeled_as_readable():
    result = mod.run()
    rough = result["longest_rough_track"]
    assert rough["audit"]["letters"] == 106
    assert rough["readability_track"] == "rough_exact_draft_not_reader_ready"
    assert rough["seam_debt"]
    assert result["reader_gate"].startswith("closed")
