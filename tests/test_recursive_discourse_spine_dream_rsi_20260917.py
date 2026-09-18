import importlib.util
import sys
from pathlib import Path

root = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "recursive_discourse_spine",
    root / "experiments" / "recursive_discourse_spine_dream_rsi_20260917.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_recursive_spine_reaches_all_requested_target_bands_with_prose():
    result = module.run()
    assert result["stats"]["rendered"] == 4
    assert result["stats"]["longest_letters"] >= 300
    assert result["stats"]["targets_reached_or_exceeded"] == 4
    for row in result["rendered_candidates"]:
        assert row["provenance"]["recursive_typed_spine"]
        assert row["provenance"]["catalogue_used"] is False
        assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]


def test_recursive_spine_is_not_a_palindrome_claim_and_has_next_repair():
    result = module.run()
    assert result["stats"]["exact"] == 0
    assert result["novelty_preflight"]["finite_scene_bank"] is False
    assert "first unresolved" in result["next_repair"]["operator"]
