import hashlib
import json
from pathlib import Path

from experiments.asynchronous_cross_clause_genitive_product_20260922 import (
    CARRIERS,
    CYCLES,
    GENITIVES,
    character_trace,
    contraction_gate,
    run,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "asynchronous_cross_clause_genitive_product_20260922.py"
ARTIFACT = ROOT / "runs" / "asynchronous-cross-clause-genitive-product-20260922.json"


def test_declared_domain_has_only_live_s_equations_and_no_saw_a_replay():
    assert all(pair.equation_holds() for pair in CARRIERS)
    assert all(pair.equation_holds() for pair in CYCLES.values())
    assert all(item.equation_holds() for item in GENITIVES)
    inventory = " ".join(
        word for pair in (*CARRIERS, *CYCLES.values()) for word in pair.left + pair.right
    )
    assert "saw" not in inventory


def test_is_and_has_are_letter_changing_controls_not_possessives():
    assert any(item.surface == "snoop's" for item in GENITIVES)
    for expansion in ("is", "has"):
        gate = contraction_gate("snoop", expansion)
        assert gate["passes"] is False
        assert gate["surface_tape"] != gate["expanded_tape"]
        assert gate["reason"] == "is/has expansion changes letters"


def test_head_attachment_does_not_pop_later_clause_debt():
    payload = run()
    obstruction = payload["first_dependency_or_residual_obstruction"]
    debt = obstruction["dependency"]
    assert debt["state"]["head_attached"] is True
    assert debt["state"]["clauses_finished"] == (True, False)
    assert debt["state"]["required_coreferent"] == "it"
    assert debt["state"]["required_finite_surface"] == "stops"
    assert debt["state"]["observed_predicate_surface"] == "stop"
    assert debt["first_undischarged"] == "clause_2_coreferential_subject"
    assert obstruction["character_register_terminal"]["closed"] is True
    assert obstruction["character_register_terminal"]["residual"] == ""


def test_bounded_operator_stops_with_long_exact_paths_but_no_survivor():
    payload = run()
    assert payload["stats"]["exact_paths_over_44"] > 0
    assert payload["stats"]["max_exact_letters"] > 44
    assert payload["stats"]["survivors"] == 0
    assert payload["survivors"] == []
    assert payload["fixed_domain"]["fragments_admitted"] is False
    assert payload["fixed_domain"]["post_hoc_repair"] is False
    assert payload["verdict"] == "stop_family_no_complete_connected_exact_survivor"


def test_every_certificate_keeps_exact_cursors_freshness_and_masks_live():
    payload = json.loads(ARTIFACT.read_text())
    assert payload["provenance"]["host"] == "hst-bench"
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == payload["provenance"]["source_sha256"]
    for row in payload["audited_certificates"]:
        audit = row["independent_audit"]
        assert audit["two_pointer_exact"] and audit["hashes_agree"]
        register = row["character_register"]
        assert register["closed"]
        assert register["owner"] == register["residual"] == ""
        assert register["left_cursor"] == register["right_cursor"]
        assert all(step["masks_live"] for step in register["trace"])
        assert "all_distinct" in row["global_lemma_freshness"]
        assert "passes" in row["complementary_boundary_mask"]
        assert "passes" in row["proper_span_mask"]


def test_character_trace_reports_nonempty_owner_until_opposite_side_returns():
    trace = character_trace(("we", "spot", "snoops'"), ("spoon", "stop", "sew"))
    assert trace["trace"][2]["owner"] == "L"
    assert trace["trace"][2]["residual"]
    assert trace["closed"]
    assert trace["left_cursor"] == trace["right_cursor"] == 12
