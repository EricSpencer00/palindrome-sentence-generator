from pathlib import Path

from experiments.validate_experiment_novelty_20260915 import validate


def test_registered_experiments_have_unique_signatures_and_artifacts():
    result = validate()
    assert result["missing"] == []
    assert result["entries"] == result["unique_signatures"]
    assert result["entries"] == result["unique_artifacts"]


def test_latest_experiments_are_registered_as_distinct_families():
    path = Path(__file__).parents[1] / "docs" / "experiment-novelty-registry.json"
    ids = {row["id"] for row in __import__("json").loads(path.read_text())["entries"]}
    assert "brown-coreferent-variable-pp" in ids
    assert "event-frame-independent-relexicalization" in ids
    assert "fresh-paired-clause-ledger" in ids
    assert "dialogue-shared-topic-elliptical-residual" in ids
    assert "morphological-derivational-seam" in ids
    assert "cp-semantic-grammar-palindrome" in ids
    assert "clause-lattice-joint-dp" in ids
    assert "character-clause-fst-joint-emission" in ids
    assert "two-bank-word-equation-seam-dp" in ids
    assert "seam-first-complete-clause-authoring" in ids
    assert "semantic-dependency-outside-in" in ids
