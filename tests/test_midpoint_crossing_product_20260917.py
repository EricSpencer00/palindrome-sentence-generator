import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_midpoint_product_matches_oracle_on_variable_length_fixtures():
    module = load("midpoint", "experiments/midpoint_crossing_product_20260917.py")
    for tokens in module.tiny_fixtures():
        text = " ".join(tokens)
        assert module.crossing_product(tokens)["closed"] == module.oracle(text)


def test_recursive_product_can_reach_descending_right_cursor_completion():
    module = load("recursive", "experiments/recursive_discourse_frame_product_20260917.py")
    assert module.product(("ab", "c"), ("cb", "a"))[0] is True


def test_scene_seam_reports_true_for_unequal_clause_partition():
    module = load("scene", "experiments/scene_seam_csp_boundary_20260917.py")
    result = module.seam("ab c", "cb a")
    assert result["closed"] is True
    assert result["first_debt"] is None
