from experiments.dream_rsi_model_guided_span_resynthesis_20260918 import (
    SEED_LEFT,
    SEED_RIGHT,
    audit,
    build_candidate,
    letters,
    segment_reverse,
)


def test_independent_audit_and_bootstrap_seed():
    seed = f"{SEED_LEFT} {SEED_RIGHT}"
    result = audit(seed)
    assert result["letters"] == 38
    assert result["two_pointer_exact"]
    assert result["sha_equal_under_reversal"]


def test_reflected_span_is_exact_without_reversing_finished_tape():
    proposal = {
        "raw_model_text": "a quiet archivist records notes",
        "span": "a quiet archivist records notes",
        "prompt": "test prompt",
        "model": "test-policy",
    }
    row = build_candidate(proposal)
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["letters"] > 38
    assert row["provenance"]["mirrored_interval_only"]
    assert not row["provenance"]["whole_tape_reversed"]
    assert letters(row["generated_span"])[::-1] == row["reflected_residual_tape"]


def test_reverse_chart_rejects_unlexicalized_residual_instead_of_certifying_it():
    surface, metadata = segment_reverse("qzxqzx")
    assert surface == ""
    assert metadata["solved"] is False
