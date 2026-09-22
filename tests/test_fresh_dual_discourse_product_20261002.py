import json
from pathlib import Path

from experiments.fresh_dual_discourse_product_20261002 import OUT, REMOTE_ORIGIN, run


def test_strict_dual_discourse_product_closes_bounded_endpoint_lane():
    result = run()
    assert result["stats"] == {
        "semantic_frames": 38,
        "unequal_grammar_products": 20,
        "states": 724,
        "transitions": 704,
        "caps_reached": 0,
        "exact_rendered_candidates": 0,
    }
    assert result["exact_candidates"] == []
    assert result["obstruction"]["kind"] == "typed_endpoint_classes_share_no_complete_lexical_arc"
    assert result["provenance"]["lane_closed"] is True
    assert result["provenance"]["remote_origin"] == REMOTE_ORIGIN


def test_committed_strict_artifact_replays_stats_and_remote_digests():
    artifact = json.loads(Path(OUT).read_text())
    replay = run()
    assert artifact["stats"] == replay["stats"]
    assert artifact["deepest_frontiers"] == replay["deepest_frontiers"]
    assert artifact["provenance"]["remote_origin"]["source_sha256"] == (
        "6094596638758ec1b6082699091a7519c7f29dbb65224a4d77406e1f34b330cb"
    )
    assert artifact["provenance"]["remote_origin"]["result_sha256"] == (
        "9a07d2ec957034114a6cd6e4f5ada87b6e7da49eae6e2cee60fafe4db3843a41"
    )
