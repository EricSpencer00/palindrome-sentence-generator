import json

from experiments.feature_constrained_live_product_20260917 import OUT, run


def test_live_slot_product_recovers_only_the_seed_control():
    result = run()
    assert set(result["cells"]) == {"seed_like", "determiner_object", "adjective", "singular_object"}
    seed = result["cells"]["seed_like"]
    assert seed["exact_paths"] == 1
    assert seed["rows"][0]["rendered"].lower() == "an aide rips nine memos; some men inspire diana"
    assert seed["rows"][0]["audit"]["sha256"] == seed["rows"][0]["audit"]["reverse_sha256"]
    assert result["novel_exact_candidates"] == []


def test_product_rows_are_live_and_dead_frontiers_are_preserved():
    result = run()
    for cell in result["cells"].values():
        assert cell["states"] > 0
        assert cell["dead_frontiers"]
        for row in cell["rows"]:
            assert row["provenance"]["live_character_product"]
            assert row["provenance"]["posthoc_reversal"] is False
            assert row["audit"]["two_pointer_exact"] == row["audit"]["exact"]


def test_run_artifact_is_the_feature_constrained_method():
    result = json.loads(OUT.read_text())
    assert result["method"] == "live slot-trie outside-in product with agreement and valency banks"
    assert result["next_repair"]["no_rlaif"] is True
