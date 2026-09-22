import json

from experiments.incumbent_666_obligation_first_phrase_lattice_20260922 import (
    OUT,
    PARENT_SHA256,
    independent_audit,
    normalize,
)


def test_obligation_first_lattice_preserves_exact_parent_and_bound():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "obligation-first-phrase-lattice-no-closure-666")

    assert row["parent_sha256"] == PARENT_SHA256
    assert row["promotion_status"]["promoted"] is False
    assert row["growth_over_parent"] == 0
    result = independent_audit(row["rendered"])
    assert result["normalized_letters"] == 666
    assert result["two_pointer_exact"]
    assert result["sha256_forward"] == PARENT_SHA256
    assert payload["working_incumbent"]["letters"] == 568
    assert len(payload["preserved_frontier"]) == 4

    attempt = row["primary_attempt"]
    assert attempt["normalized_windows"] == {"normalized_left": [197, 281], "normalized_right": [385, 469]}
    assert attempt["raw_windows"] == {"raw_left": [263, 384], "raw_right": [520, 644]}
    assert attempt["equation"] == {"left_length": 84, "right_length": 84, "right_is_reverse": True}
    assert attempt["required_reverse_prefix"]["left_obligation_derived_from_right"].startswith("nownoeldidilive")
    assert attempt["required_reverse_prefix"]["right_obligation_derived_from_left"].startswith("stressedwasiaron")
    for side in ("left_lattice", "right_lattice"):
        lattice = attempt[side]
        assert lattice["beam_width"] == 16
        assert lattice["max_expansions"] == 128
        assert lattice["max_clauses_per_side"] == 4
        assert lattice["max_backtracks_per_cursor"] == 2
        assert lattice["expansions"] <= 128
        assert lattice["exact_closure"] is False
        obstruction = lattice["obstruction"]
        assert {"cursor", "expected", "emitted", "residual", "owner", "reason", "grammar_state"} <= obstruction.keys()
        for key in (
            "subject_stack",
            "object_stack",
            "sentence_phase",
            "punctuation",
            "active_discourse_entity",
            "pending_causal_temporal_relation",
            "used_clause_ids",
            "used_frame_set",
            "neighboring_boundary_context",
        ):
            assert key in obstruction["grammar_state"]
    assert attempt["left_lattice"]["obstruction"]["cursor"] == 38
    assert attempt["left_lattice"]["obstruction"]["reason"] == "disconnected_catalogue_clause"
    assert attempt["left_lattice"]["obstruction"]["residual"].startswith("patnotes")
    assert attempt["right_lattice"]["obstruction"]["cursor"] == 28
    assert attempt["right_lattice"]["obstruction"]["reason"] == "disconnected_catalogue_clause"
    assert attempt["right_lattice"]["obstruction"]["residual"].startswith("dogwasaram")
    assert attempt["closures"] == []
    assert attempt["exact_child_saved"] is False


def test_lattice_switches_once_to_different_actual_seam_and_persists_residuals():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "obligation-first-phrase-lattice-no-closure-666")
    attempt = row["switched_attempt"]

    assert attempt["switch_reason"] == "immediate switch after primary seam obstruction"
    assert attempt["normalized_windows"] == {"normalized_left": [127, 197], "normalized_right": [469, 539]}
    assert attempt["raw_windows"] == {"raw_left": [171, 263], "raw_right": [644, 736]}
    assert attempt["equation"]["right_is_reverse"] is True
    assert attempt["equation"]["left_length"] == attempt["equation"]["right_length"] == 70
    assert attempt["left_lattice"]["obstruction"]["cursor"] == 43
    assert attempt["left_lattice"]["obstruction"]["reason"] == "repeated_neighbor_frame"
    assert attempt["left_lattice"]["obstruction"]["expected"] == "n"
    assert attempt["left_lattice"]["obstruction"]["residual"].startswith("noraseesaram")
    assert attempt["right_lattice"]["obstruction"]["cursor"] == 15
    assert attempt["right_lattice"]["obstruction"]["reason"] == "repeated_neighbor_frame"
    assert attempt["right_lattice"]["obstruction"]["expected"] == "m"
    assert attempt["right_lattice"]["obstruction"]["residual"].startswith("maraseesaron")
    assert attempt["closures"] == []
    assert attempt["exact_child_saved"] is False
