import hashlib
import json
from pathlib import Path

from experiments.productive_affix_return_stack_20260922 import (
    FAMILIES,
    clean_extended_witness,
    complementary_boundary_mask,
)
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "productive-affix-return-stack-20260922.json"
SOURCE = ROOT / "experiments" / "productive_affix_return_stack_20260922.py"


def test_extended_witness_is_exact_clean_and_longer_than_incumbent() -> None:
    row = clean_extended_witness()
    assert row["rendered"] == "No trace. Note sleet. Spoons snoop. Steel? Set one carton."
    assert row["letters"] == 44 > 42
    assert row["normalized_tape"] == row["normalized_tape"][::-1]
    assert row["sha256_forward"] == "96462b7bc06958668e9d13d13ebd0b63e4f682a51939c5bc34d7aa230d39b104"
    assert row["sha256_forward"] == row["sha256_reverse"]
    assert row["derivation"]["carrier_equation"]
    assert all(row["derivation"]["cycle_equations"])
    assert row["morphology"]["lemma"] == "spoon"
    assert row["morphology"]["surface"] == "spoons"
    assert row["morphology"]["agreement"] == "plural subject"
    assert row["lemma_freshness"]["all_distinct"]
    assert row["complementary_boundary_mask"]["passes"]
    assert all(row["mechanical_checks"].values())


def test_return_phrases_are_flushed_in_strict_lifo_order() -> None:
    trace = clean_extended_witness()["stack_trace"]
    pushed = [tuple(step["right"]) for step in trace if step["operation"].startswith("push")]
    popped = [tuple(step["right"]) for step in trace if step["operation"] == "pop_return"]
    assert pushed == [("set", "one", "carton"), ("steel",), ("snoop",)]
    assert popped == list(reversed(pushed))


def test_complementary_mask_rejects_internal_finished_span() -> None:
    assert complementary_boundary_mask(
        ("no", "trace", "note", "sleet", "spoons"),
        ("snoop", "steel", "set", "one", "carton"),
    )["passes"]
    assert not complementary_boundary_mask(("stops", "live"), ("evil", "spots"))["passes"]


def test_artifact_covers_every_requested_family_and_records_exact_obstructions() -> None:
    payload = json.loads(ARTIFACT.read_text())
    assert [row["family"] for row in payload["families"]] == [row["family"] for row in FAMILIES]
    by_family = {row["family"]: row for row in payload["families"]}
    assert by_family["past_participle_ed"]["domains"] == {
        "carriers": 37, "cycles": 0, "productive_inner_cycles": 0,
    }
    assert by_family["progressive_ing"]["obstruction_or_gate"]["cursor"] == {
        "side": "right", "required_prefix": "gni", "domain_size": 0,
    }
    assert by_family["comparative_er"]["domains"]["cycles"] == 1
    assert by_family["comparative_er"]["domains"]["productive_inner_cycles"] == 0


def test_artifact_survivors_are_independently_reaudited() -> None:
    payload = json.loads(ARTIFACT.read_text())
    assert payload["provenance"]["host"] == "hst-bench"
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == payload["provenance"]["source_sha256"]
    assert payload["survivors"]
    for row in payload["survivors"]:
        normalized = normalize_letters(row["rendered"])
        assert normalized == row["normalized_tape"]
        assert normalized == normalized[::-1]
        assert hashlib.sha256(normalized.encode()).hexdigest() == row["sha256_forward"]
        assert row["complementary_boundary_mask"]["passes"]
        assert all(mechanical_admission_checks(row["rendered"], min_letters=43, max_letters=240).values())
