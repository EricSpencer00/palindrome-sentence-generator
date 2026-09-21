import json
from pathlib import Path


def test_two_character_outer_classes_use_reversed_suffix_orientation():
    data = json.loads(
        Path("runs/two-character-outer-class-csp-20260921.json").read_text()
    )
    assert data["stats"] == {
        "outer_states_pruned": 162,
        "retained_outer_states": 81,
        "lexical_prunes": 0,
        "rendered_controls": 81,
        "center_seam_misses": 81,
        "exact_gt38": 0,
        "max_letters": 82,
    }
    assert data["novelty_preflight"]["status"] == "passed"
    assert data["provenance"]["audits"] == [
        "independent full-tape pointer",
        "independent SHA-256 forward/reverse",
    ]
    for row in data["diagnostic_controls"]:
        choice = row["outer_choice"]
        assert choice["prefix_2"] == choice["suffix_2_reversed"]
        assert row["provenance"]["reader_certification"] is False
        assert row["audit"]["pointer_exact"] is False

