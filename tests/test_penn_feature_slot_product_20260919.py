import json
from pathlib import Path


def test_penn_feature_slot_product_is_feature_conditioned_and_exact_gated():
    data = json.loads(Path("runs/penn-feature-slot-product-20260919.json").read_text())
    assert data["provenance"]["penn_tags"]
    assert data["provenance"]["word_feature_maps"]
    assert data["provenance"]["aligned_token_mirror"] is False
    assert sum(item["exact"] for item in data["feature_stats"].values()) == 0
    assert data["candidates"] == []
    assert set(data["feature_stats"]) == {"singular_vbz", "plural_vbp", "past_vbd"}
    assert data["feature_stats"]["singular_vbz"]["states"] == 51
    assert data["feature_stats"]["plural_vbp"]["states"] == 21
    assert data["feature_stats"]["past_vbd"]["states"] == 41
    assert data["adjunct_frame_unique"] == 13650
