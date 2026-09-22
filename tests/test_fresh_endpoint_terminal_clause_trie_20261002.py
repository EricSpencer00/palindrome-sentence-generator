import hashlib
import json
from pathlib import Path

from experiments.fresh_endpoint_terminal_clause_trie_20261002 import OUT, run


def test_endpoint_trie_stops_before_predicate_at_exact_determiner_conflict():
    result = run()
    assert result["seed"] == {
        "matched_prefix": "anera",
        "matched_letters": 5,
        "owner": "left",
        "residual": "ser",
        "left_boundaries": [2, 8],
        "right_exposed_boundaries": [5],
        "complementary_boundaries": [],
        "next_left_predicates": ["erases", "removes"],
        "next_right_required_tokens": ["an"],
    }
    assert result["stats"]["predicate_unification_attempts"] == 0
    assert result["stats"]["exact_rendered_candidates"] == 0
    assert result["obstruction"]["first_conflict"] == {"left": "s", "right": "n"}
    assert result["obstruction"]["matched_cursor_zero_based"] == 5
    assert result["provenance"]["lane_closed"] is True


def test_endpoint_artifact_has_replayable_source_and_remote_provenance():
    artifact = json.loads(Path(OUT).read_text())
    replay = run()
    source = Path(__file__).resolve().parents[1] / "experiments" / "fresh_endpoint_terminal_clause_trie_20261002.py"
    assert artifact["seed"] == replay["seed"]
    assert artifact["obstruction"] == replay["obstruction"]
    assert artifact["provenance"]["repo_source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert artifact["provenance"]["remote_origin"]["source_sha256"] == (
        "a6f44b33ed39316493d7bd8b8b47d8ca4daef1fc4ebe0a4dd078f486767fdf33"
    )
    assert artifact["provenance"]["remote_origin"]["result_sha256"] == (
        "8aad9a5d5b5485170a4eaba9c9597b20833a5d6ee9bc75e190df0b369807e535"
    )
