import hashlib
import json
from pathlib import Path

from experiments.exact_by_construction_clause_product_20260917 import (
    OUT,
    build_clause_grammar,
    exact_product,
    independent_audit,
    run,
)


def test_product_matches_a_withheld_exact_control_without_using_it_as_output():
    result = run(max_states=300_000)
    assert result["withheld_control"]["exact_paths"] == 1
    assert result["withheld_control"]["not_a_generated_candidate"] is True
    assert result["provenance"]["seed_used_in_output"] is False


def test_fresh_grammar_is_complete_clause_first_and_independently_audited():
    result = json.loads(OUT.read_text())
    assert result["method"] == "exact_by_construction_optional_slot_clause_product"
    assert result["grammar"]["independent_word_boundaries"] is True
    assert result["grammar"]["complete_clause_only"] is True
    assert result["search"]["rlaif_per_candidate"] is False
    for row in result["exact_candidates"]:
        assert row["independent_audit"]["exact"]
        assert row["independent_audit"]["algorithm"] == "independent_two_pointer"
        assert row["independent_parse"]["ok"]
        assert row["rendered"].endswith(".")
    expected = hashlib.sha256(
        (Path(__file__).parents[1] / "experiments/exact_by_construction_clause_product_20260917.py").read_bytes()
    ).hexdigest()
    assert result["provenance"]["generator_sha256"] == expected
    assert result["next_repair"]["operator"] == "add_independent_clause_attachment"


def test_product_has_no_shortcut_generated_path_in_current_run():
    result = json.loads(OUT.read_text())
    assert all(not row["anti_shortcut"]["word_order_mirror"] for row in result["exact_candidates"])
    assert all(not row["anti_shortcut"]["self_palindromic_spans"] for row in result["exact_candidates"])
