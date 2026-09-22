import json
from pathlib import Path

from experiments.semantic_pivot_cfg_20261001 import OUT, run


def test_pivot_bank_has_no_reversed_lexical_pairs():
    result = run()
    assert result["bank_checks"]["reverse_pair_free"]
    assert result["provenance"]["independent_audits"] == ["two-pointer", "forward/reverse SHA-256"]


def test_every_reported_candidate_has_independent_exact_audit():
    result = run()
    for row in result["exact_candidates"]:
        assert row["independent_audit"]["two_pointer_exact"]
        assert row["independent_audit"]["sha_equal"]


def test_artifact_matches_run_shape():
    payload = json.loads(Path(OUT).read_text())
    assert payload["experiment_id"] == "semantic-pivot-cfg-20261001"
    assert payload["reader_gate"].startswith("closed")
