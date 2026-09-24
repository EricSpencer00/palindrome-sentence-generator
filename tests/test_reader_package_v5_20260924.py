from experiments.reader_package_v5_20260924 import build
from training.dpo_readability_judge import prepare


def test_blinded_frontier_packet_is_deterministic_and_unranked() -> None:
    first, second = build(), build()
    assert first["rater_form"] == second["rater_form"]
    assert first["answer_key"] == second["answer_key"]
    assert len(first["rater_form"]["items"]) == 14
    assert first["reader_protocol"]["length_is_not_a_quality_score"]
    assert first["reader_protocol"]["human_ratings_collected"] is False
    assert first["reader_protocol"]["programmatic_metrics_certify_readability"] is False
    assert {r["letters"] for r in first["answer_key"]["candidates"]} == {38, 650, 654, 666}
    assert all(r["audit"]["two_pointer"]["two_pointer_exact"]
               and r["audit"]["second_project_audit"]["exact"]
               for r in first["answer_key"]["candidates"])


def test_rater_form_hides_provenance_and_includes_both_control_types() -> None:
    packet = build()
    form_text = str(packet["rater_form"])
    assert "candidate_vs_candidate" not in form_text
    assert "outer-scene-654" not in form_text
    assert all(item["a"]["text"] != item["b"]["text"]
               for item in packet["rater_form"]["items"])
    kinds = {task["type"] for task in packet["answer_key"]["tasks"]}
    assert kinds == {"candidate_vs_shuffle", "intact_vs_shuffle", "candidate_vs_candidate"}
    assert packet["reproducibility"]["pair_count"] == 14


def test_no_uncollected_preferences_are_reported_as_dpo_readiness() -> None:
    dataset = prepare(build(), {"raters": []})
    assert dataset["status"] == "insufficient_reader_preferences"
    assert dataset["counts"]["preference_examples"] == 0
    assert dataset["training_data"] == []
    assert dataset["validation_data"] == []
