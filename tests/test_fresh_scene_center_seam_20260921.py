import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("fresh_scene", ROOT / "experiments/fresh_scene_center_seam_20260921.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def test_fresh_scene_uses_online_seam_and_unequal_boundaries():
    result = mod.run()
    assert result["stats"]["candidates"] == 8
    assert result["stats"]["unequal_boundary_controls"] == 8
    assert result["stats"]["exact"] == 0
    for row in result["rendered_controls"]:
        assert row["provenance"]["render_after_equation"]
        assert row["gates"]["exact_independent_audits"]
        assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]
        assert row["left_clause"] != row["right_clause"]
    assert "three-character bridge" in result["next_operator"]
