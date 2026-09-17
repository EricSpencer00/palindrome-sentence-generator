import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("seam", Path(__file__).parents[1] / "experiments/cross_boundary_seam_lattice_20260917.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def test_seam_lattice_recovers_seed_only_as_control_and_audits_every_row():
    out = mod.run()
    assert out["stats"]["products"] == len(out["rows"])
    assert out["stats"]["seed_controls"] == 1
    assert out["stats"]["reader_eligible"] == 0
    assert out["status"] == "quarantined_posthoc_comparison"
    assert out["search_integrity"]["live_product_search"] is False
    seed = next(r for r in out["rows"] if r["provenance"]["seed_regression_control"])
    assert seed["audit"]["exact"] and seed["normalized_length"] == 38
    assert seed["seam_segmentation"]["variable_boundaries"]
    for row in out["rows"]:
        assert row["audit"]["two_pointer_exact"] == row["audit"]["exact"]
        assert row["audit"]["sha256"] != row["audit"]["reverse_sha256"] or row["audit"]["exact"]
        assert row["provenance"]["generated_not_catalogue"]
        assert row["provenance"]["posthoc_comparison"]
        assert row["next_repair"]
