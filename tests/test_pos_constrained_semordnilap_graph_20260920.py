import importlib.util
from pathlib import Path

P = Path(__file__).parents[1] / "experiments/pos_constrained_semordnilap_graph_20260920.py"
spec = importlib.util.spec_from_file_location("posgraph", P)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def test_pos_graph_has_independent_prose_controls_and_audits():
    result = mod.run()
    assert result["stats"]["controls"] == 32
    assert result["stats"]["exact_gt38"] == 0
    for row in result["diagnostic_controls"]:
        assert row["audit"]["two_pointer_checked"]
        assert row["anti_shortcut"]["word_order_symmetry"] is False
        assert row["anti_shortcut"]["catalogue_text"] is False
        assert row["anti_shortcut"]["repeated_units"] is False

