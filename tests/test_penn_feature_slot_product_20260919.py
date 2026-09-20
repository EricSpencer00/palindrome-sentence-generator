import json
from pathlib import Path


def test_penn_feature_slot_product_is_feature_conditioned_and_exact_gated():
    data = json.loads(Path("runs/penn-feature-slot-product-20260919.json").read_text())
    assert data["provenance"]["penn_tags"]
    assert data["provenance"]["word_feature_maps"]
    assert data["provenance"]["adjunct_partition_audit"] == "frame-attested DET-NOUN-VERB-ADP-NOUN"
    assert data["provenance"]["aligned_token_mirror"] is False
    assert sum(item["exact"] for item in data["feature_stats"].values()) == 0
    assert data["candidates"] == []
    assert set(data["feature_stats"]) == {"singular_vbz", "plural_vbp", "past_vbd"}
    assert all(item["states"] >= 0 for item in data["feature_stats"].values())
    assert data["adjunct_frame_unique"] > 0
    assert data["frame_identity_counts"]
    assert data["feature_stats"]["singular_vbz"]["states"] == 34
    assert data["feature_stats"]["plural_vbp"]["states"] == 15
    assert data["feature_stats"]["past_vbd"]["states"] == 33
    assert all(item["object_words"] > 0 for item in data["feature_stats"].values())
