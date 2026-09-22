import importlib.util
from pathlib import Path


P = Path(__file__).parents[1] / "experiments/vocative_abba_clause_search_20260927.py"
spec = importlib.util.spec_from_file_location("vocative_abba", P)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_abba_paragraph_is_exact_and_novel():
    result = mod.run()
    best = result["best"]
    assert best is not None
    assert best["letters"] == 68
    assert best["audit"]["two_pointer_exact"]
    assert best["audit"]["validator_exact"]
    assert best["audit"]["sha_equal"]
    assert best["novelty_preflight"]
    assert best["provenance"]["distinct_units"]
    assert not best["provenance"]["self_palindromic_units"]
    assert not best["provenance"]["finished_tape_reversal"]


def test_reader_package_is_seeded_and_keeps_exactness_out_of_rating():
    result = mod.run()
    package = result["reader_package"]
    assert package["random_seed"] == 20260927
    assert package["status"].startswith("prepared")
    assert "exactness is not a rating dimension" in package["instructions"]
    assert len(package["order"]) == len(package["items"])
