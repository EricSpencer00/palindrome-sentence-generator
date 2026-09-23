from experiments.seed_np_typed_possessive_head_relative_20260922 import (
    CONTROL_SHA256,
    TYPED_PRODUCTIONS,
    build_payload,
    independent_audit,
)


def test_control_and_typed_production_bound_are_preserved():
    result = build_payload()
    assert result["control"]["audit"]["letters"] == 54
    assert result["control"]["audit"]["sha256_forward"] == CONTROL_SHA256
    assert result["control"]["audit"]["two_pointer_exact"]
    assert len(TYPED_PRODUCTIONS) == 8
    assert result["stats"]["typed_production_count"] == 8
    assert result["preserved_frontiers"]["568"]["letters"] == 568
    assert result["preserved_frontiers"]["666"]["letters"] == 666


def test_possessive_rows_record_independent_obstruction_and_gates():
    result = build_payload()
    assert result["stats"]["exact_closures"] == 0
    assert result["operator_change"]["changed_within_run"] is True
    for row in result["rows"]:
        assert row["length"] > 54
        assert row["independent_exact_audit"] == independent_audit(row["rendered"])
        assert row["independent_exact_audit"]["two_pointer_exact"] is False
        assert row["online_join"]["closed"] is False
        assert row["online_join"]["first_mismatch"]
        assert row["online_join"]["residual"]
        assert row["online_join"]["inherited_residual"] == "m"
        assert row["online_join"]["inherited_residual_consumed"] is False
        assert row["project_lexicon_gate"]["all_project_lexicon"] is True
        assert row["project_lexicon_gate"]["no_new_self_palindromic_word"] is True
        assert row["novelty_shortcut_gates"]["no_self_palindromic_added_span"] is True
        assert row["novelty_shortcut_gates"]["no_repeated_content_shortcut"] is True
        assert row["provenance"]["center_event_pair_reused"] is False
        assert row["reader_status"] == "not_certified; exact closure required"
