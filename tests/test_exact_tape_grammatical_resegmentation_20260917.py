import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "resegmentation", Path(__file__).parents[1] / "experiments/exact_tape_grammatical_resegmentation_20260917.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_bounded_resegmentation_renders_candidates_and_reports_zero_exact():
    result = mod.run()
    assert result["stats"]["rendered"] == 3
    assert result["stats"]["exact_count"] == 0
    assert all(row["audit"]["right_resegmentation_grammar"] for row in result["candidates"])
    assert all(row["audit"]["two_pointer_exact"] is False for row in result["candidates"])
    assert all(row["audit"]["normalized_sha256"] for row in result["candidates"])
    assert result["novelty_preflight"]["fixed_tape_used"] is False


def test_reverse_dp_is_bounded_and_uses_variable_boundaries():
    words, explored, status = mod.segment_reverse(mod.tape(mod.LEFT[0])[::-1])
    assert explored <= mod.MAX_STATES
    assert status in {"complete", "state_limit"}
    assert isinstance(words, list)
