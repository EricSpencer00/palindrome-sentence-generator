from experiments.incumbent_568_outer_event_resegmentation_20260923 import build_payload
from llm_palindrome.validator import is_palindrome


def test_outer_event_resegmentation_is_an_exact_568_lineage_child():
    payload = build_payload()
    row = payload["rows"][0]
    assert payload["parent"]["letters"] == 568
    assert payload["parent"]["sha256"] == "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
    assert row["letters"] == 574
    assert row["parent_edit"]["growth_over_parent"] == 6
    assert row["audit"]["two_pointer_exact"] is True
    assert row["audit"]["project_validator_exact"] is True
    assert row["audit"]["sha_equal"] is True
    assert is_palindrome(row["rendered"])


def test_live_character_obligation_closes_without_changing_the_retained_middle():
    row = build_payload()["rows"][0]
    seam = row["live_seam"]
    assert seam["equation"]["left"] == "irisspotsaratarisawaram"
    assert seam["equation"]["left"] == seam["equation"]["right_obligation"]
    assert seam["characters_consumed"] == 23
    assert seam["final_residual"] == ""
    assert all(item["consumed"] for item in seam["cursor_trace"])
    assert row["parent_edit"]["retained_middle_unchanged"] is True
    assert row["construction_debt"]["reader_evidence"] is False


def test_candidate_records_clause_boundary_debt_instead_of_claiming_readability():
    row = build_payload()["rows"][0]
    boundary = row["construction_debt"]["boundary_audit"]
    assert boundary["left_tokens"] == ["Iris", "spots", "a", "rat", "Ari", "saw", "a", "ram"]
    assert boundary["right_tokens"] == ["Mara", "was", "Ira", "Tara", "stops", "Siri"]
    assert boundary["shared_reflected_boundaries"]
    assert "not established" in row["construction_debt"]["discourse_coherence"]


def test_tokenwise_exact_alternative_is_preserved_but_rejected():
    proposal = build_payload()["rejected_proposals"][0]
    assert proposal["letters_if_spliced"] == 580
    assert proposal["local_equation_exact"] is True
    assert len(proposal["rejection_reasons"]) == 2
    assert "tokenwise shortcut" in proposal["rejection_reasons"][0]
    assert "remaining-shell-global-gate" in proposal["rejection_reasons"][1]


def test_outer_shell_repair_improves_length_and_uses_non_tokenwise_equation():
    payload = build_payload()
    repaired = payload["rows"][-1]
    assert repaired["id"] == "outer-event-resegmentation-repair-578"
    assert repaired["letters"] == 578
    assert repaired["normalized_sha256"] == "ced52c6e23a246bf83fbd1698698f6b5d7e41b5f56ce27dcd059123c436dead2"
    assert repaired["parent_edit"]["growth_over_parent"] == 10
    assert repaired["audit"]["two_pointer_exact"] is True
    assert repaired["audit"]["project_validator_exact"] is True
    assert repaired["audit"]["sha_equal"] is True
    assert payload["best_new_exact_child"]["id"] == repaired["id"]
    assert repaired["construction_debt"]["reader_evidence"] is False


def test_failed_outer_expansion_changes_the_seam_instead_of_repeating_the_same_bank():
    payload = build_payload()
    rejected = payload["rejected_proposals"][1]
    assert rejected["letters_per_surface"] == 52
    assert rejected["local_equation_exact"] is True
    assert rejected["full_child_admitted"] is False
    assert "repeats the saw/a-ram event frame" in rejected["rejection_reasons"][0]
    assert "[20,48)/[520,548)" in payload["next_operator"]
