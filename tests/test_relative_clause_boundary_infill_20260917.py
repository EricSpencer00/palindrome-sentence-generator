import importlib.util
from pathlib import Path

root = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "relative_clause_boundary_infill",
    root / "experiments" / "relative_clause_boundary_infill_20260917.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_boundary_crossing_route_renders_complete_authored_prose():
    result = module.run()
    assert result["stats"]["rendered"] == 18
    assert result["stats"]["longest_letters"] > 70
    assert len({row["rendered"] for row in result["rendered_candidates"]}) == 18
    for row in result["rendered_candidates"]:
        assert row["provenance"]["relative_clause_boundary_infill"]
        assert row["provenance"]["catalogue_used"] is False
        assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]
        assert row["boundary_crossing"]["word_boundaries_reopened"]


def test_route_records_novelty_and_concrete_next_geometry():
    result = module.run()
    assert result["stats"]["exact"] == 0
    assert result["novelty_preflight"]["prior_route_reused"] is False
    assert result["next_repair"]["route_exhausted"] is True
    assert "three-region" in result["next_repair"]["operator"]
