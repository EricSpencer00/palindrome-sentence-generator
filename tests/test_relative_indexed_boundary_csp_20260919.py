from experiments.relative_indexed_boundary_csp_20260919 import run, search


def test_relative_path_records_provenance_without_language_model():
    result = run(range(55, 65), max_nodes=20_000)
    assert result["stats"]["nodes"] > 0
    assert result["provenance"]["rlaif_per_candidate"] is False
    assert result["novelty_preflight"]["status"] == "passed"


def test_relative_path_enforces_agreement_before_seam():
    result = search(50, max_nodes=20_000)
    for row in result["actual_candidates"]:
        words = row["word_path"]
        # Every generated row has a complete relative path; no detached
        # relative verb can appear without its relative subject.
        assert len(words) == 7
