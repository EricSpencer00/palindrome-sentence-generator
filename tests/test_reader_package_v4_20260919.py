from experiments.reader_package_v4_20260919 import build


def test_package_is_deterministic_and_blinded():
    first = build()
    second = build()
    assert first["rater_form"] == second["rater_form"]
    assert first["answer_key"] == second["answer_key"]
    assert first["reader_protocol"]["randomized_blinded_order"] is True
    assert first["reader_protocol"]["programmatic_metrics_certify_readability"] is False


def test_every_frontier_item_has_intact_and_shuffled_pair_with_audit():
    result = build()
    assert len(result["rater_form"]["items"]) == 5
    for item in result["rater_form"]["items"]:
        assert item["a"]["text"] != item["b"]["text"]
    assert any(row["condition"] == "intact" and row["audit"]["exact"] for row in result["answer_key"])
    assert all("mechanical_checks" in row for row in result["answer_key"])
