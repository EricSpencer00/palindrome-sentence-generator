import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_finite_state_boundary_beam_run_is_bounded_and_audited():
    run = json.loads((ROOT / "runs/finite-state-boundary-beam-20260917.json").read_text())
    assert run["novelty_preflight"]["status"] == "passed"
    assert run["search"]["beam_width"] == 6
    assert run["search"]["paired_candidates"] == 12
    assert run["stats"]["intact_prose_controls"] > 0
    assert run["stats"]["exact"] == 0
    for candidate in run["candidates"]:
        assert candidate["audit"]["algorithm"] == "independent_two_pointer_plus_forward_reverse_sha256"
        assert candidate["provenance"]["grammar_state"].startswith("START ->")
        if candidate["failure_reason"]:
            assert candidate["next_repair"]


def test_finite_state_boundary_beam_rejects_shortcut_controls():
    run = json.loads((ROOT / "runs/finite-state-boundary-beam-20260917.json").read_text())
    assert run["rejection_controls"]["mirror_rejection"]["not_word_order_symmetry"] is False
    assert run["rejection_controls"]["fragment_rejection"]["ordinary_short_words"] is False
