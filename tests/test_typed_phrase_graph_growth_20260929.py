from experiments.typed_phrase_graph_growth_20260929 import run


def test_growth_is_exact_and_longer_than_incumbent():
    candidate = run()["candidate"]
    assert candidate["letters"] > 156
    assert candidate["exact_two_pointer"] is True
    assert candidate["validator"] is True
    assert candidate["sha256"] == candidate["independent_forward_reverse_sha256"]


def test_growth_has_distinct_units_and_online_admissions():
    result = run()
    candidate = result["candidate"]
    assert candidate["novelty_preflight"] is True
    growth = [item for item in result["growth_trace"] if item.get("edge") != "incumbent"]
    assert growth and all(item["matched_online"] and item["accepted"] for item in growth)
    assert candidate["provenance"]["posthoc_character_repair"] is False
