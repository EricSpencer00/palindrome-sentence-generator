from pathlib import Path
import json

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
    assert "synchronous-semantic-parse-equations" in ids
    assert "internal-center-window-repair" in ids
    assert "discourse-plan-coupled-expansion" in ids
    assert "collocation-synchronous-grammar" in ids
    assert "human-compositional-center-window" in ids
    assert "global-semantic-paraphrase-rewrite" in ids
    assert "collocation-graph-path" in ids
    assert "semantic-sentence-pair-alignment" in ids
    assert "template-analogy-semantic-lexicalization" in ids
    assert "neural-dual-prefix-beam-v2" in ids
    assert "grammar-intersection-chart" in ids
    assert "dependency-attribute-grammar-chart" in ids
    assert "evolutionary-prose-genome" in ids
    assert "bpe-dual-continuation" in ids
    assert "lexicalized-tag-yield-equation" in ids
    assert "mined-phrase-chunk-clause-composition" in ids
    assert "variable-boundary-tape-ilp" in ids


def test_latest_artifacts_were_preflighted_before_self_registration():
    root = Path(__file__).parents[1]
    registry_count = len(json.loads((root / "docs/experiment-novelty-registry.json").read_text())["entries"])
    for name in ("evolutionary-prose-genome-20260915.json", "dependency-attribute-grammar-chart-20260915.json", "bpe-dual-continuation-20260915.json", "lexicalized-tag-yield-equation-20260915.json"):
        audit = json.loads((root / "runs" / name).read_text())["novelty_audit"]
        # Later distinct experiments may register after this artifact was
        # produced; the preflight count must be a prior snapshot, not an
        # unstable equality with today's registry size.
        assert 0 < audit["registry_entries_read_before_run"] < registry_count
        assert not audit.get("self_entry_present", [])
        assert not audit.get("replay_of_registered_family", False)
    chart = json.loads((root / "runs" / "grammar-intersection-chart-20260915.json").read_text())
    assert 0 < chart["registry_entry_count_before_run"] < registry_count
    assert chart["prior_signatures_overlap"] == []
    phrase_run = json.loads((root / "runs" / "mined-phrase-chunk-clause-composition-v3-20260915.json").read_text())
    assert phrase_run["novelty_audit"]["registry_entries_read_before_run"] < registry_count
    assert phrase_run["novelty_audit"]["signature_overlap"] == []
    assert phrase_run["stats"]["exact_candidates"] == 0
    assert len(phrase_run["rendered_candidates_and_probes"]) == 40
    ilp_run = json.loads((root / "runs" / "variable-boundary-tape-ilp-20260915.json").read_text())
    assert ilp_run["novelty_audit"]["registry_entries_read_before_run"] < registry_count
    assert ilp_run["novelty_audit"]["prior_signatures_overlap"] == []
    assert ilp_run["solver_summary"]["exact_solution_count"] == 0
    assert ilp_run["results"][0]["rendered_probe"]["independent_exact_audit"]["exact"] is False
