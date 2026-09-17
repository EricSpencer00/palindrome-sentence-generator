import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "luna_bilateral_slot_equation_search_20260917",
    ROOT / "experiments" / "luna_bilateral_slot_equation_search_20260917.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_withheld_control_is_exact_and_long_search_has_no_false_closure():
    result = MODULE.run()
    control = result["withheld_short_control"]
    assert control["exact"] is True
    assert control["rendered"] == "step on no pets"
    assert result["exact_candidates"] == []
    assert result["status"] == "completed_no_exact_closure"


def test_residual_crosses_word_boundaries_without_posthoc_reversal():
    eq = MODULE.consume_residual("step on", "no pets")
    assert eq["matched"] == 6
    assert eq["closed"] is True
