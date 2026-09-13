import json
from pathlib import Path
import subprocess
import sys

from experiments.assisted_candidate_construction_pilot_20260913 import (
    FairProposalQueue,
    Proposal,
    Region,
    analyze_tape,
    apply_proposal,
    candidate_records,
    close_proposal,
    enumerate_segmentations,
    lexical_evidence,
    root_state,
    run_pilot,
)


def proposal(identifier, parent, operation="continue", left="ab", right="ba", **kwargs):
    return Proposal(identifier, parent, {"kind": "fixture", "label": identifier}, operation,
                    left_text=left, right_text=right, **kwargs)


def test_broad_l1_keeps_ordinary_short_function_words_without_six_letter_filter():
    evidence = lexical_evidence()
    assert {"a", "an", "in", "to", "the"} <= evidence["words"]
    assert evidence["evidence"]["safe_dictionary_count"] > 10_000
    assert "six-letter" in evidence["evidence"]["content_policy"]


def test_state_owns_all_viable_word_boundary_and_prefix_analyses():
    analyses = enumerate_segmentations("another")
    chart = analyze_tape("another")
    assert ("another",) in analyses
    assert ("an", "other") in analyses
    assert chart.complete_segmentations == len(analyses)
    assert (2, "other") in chart.open_prefixes


def test_coordinated_continuation_replays_exact_debt_without_synthesizing_right_text():
    root = root_state()
    child, event = apply_proposal(root, proposal("joint", root.state_id, left="ab", right="ba"))
    assert child is not None and event["accepted"]
    assert child.left_tape == "ab"
    assert child.right_tape == "ba"
    assert child.paired_tape == "ab"
    assert child.symmetric_debt == ""
    assert event["input"]["right_text"] == "ba"


def test_mismatched_independent_regions_are_rejected_and_parent_is_unchanged():
    root = root_state()
    child, event = apply_proposal(root, proposal("conflict", root.state_id, left="ab", right="xy"))
    assert child is None
    assert event["rejection"] == "character_conflict_between_independently_proposed_regions"
    assert root.left_tape == root.right_tape == ""


def test_bounded_coordinated_edit_can_leave_and_later_resolve_debt():
    root = root_state()
    base, _ = apply_proposal(root, proposal("base", root.state_id))
    assert base is not None
    widened = Proposal("widen", base.state_id, {"kind": "fixture"}, "reopen",
                      left_region=Region(0, 2, "abc"), right_region=Region(0, 2, "ba"))
    child, event = apply_proposal(base, widened)
    assert child is not None and event["accepted"]
    assert (child.debt_side, child.symmetric_debt) == ("left", "c")
    settle = Proposal("settle", child.state_id, {"kind": "fixture"}, "reopen",
                      left_region=Region(2, 3, "c"), right_region=Region(0, 2, "cba"))
    settled, _ = apply_proposal(child, settle)
    assert settled is not None
    assert settled.paired_tape == settled.left_tape
    assert settled.symmetric_debt == ""


def test_global_reopen_changes_an_outer_region_not_just_the_inward_frontier():
    root = root_state()
    base, _ = apply_proposal(root, proposal("base", root.state_id, left="ab", right="ba"))
    assert base is not None
    reopen = Proposal("outer-repair", base.state_id, {"kind": "fixture"}, "reopen",
                      left_region=Region(0, 1, "c"), right_region=Region(0, 2, "bc"))
    repaired, event = apply_proposal(base, reopen)
    assert repaired is not None and event["accepted"]
    assert repaired.left_tape == "cb"
    assert repaired.right_tape == "bc"
    assert repaired.symmetric_debt == ""
    assert repaired.edit_history[-1]["left_region"]["start"] == 0


def test_fair_queue_interleaves_ready_parent_regions():
    root = root_state()
    first = proposal("first", root.state_id)
    child, _ = apply_proposal(root, first)
    assert child is not None
    sibling = proposal("sibling", root.state_id, left="cd", right="dc")
    grandchild = proposal("grandchild", child.state_id, left="ef", right="fe")
    report = run_pilot([first, sibling, grandchild], state_budget=20)
    assert [row["proposal_id"] for row in report["proposal_events"]] == ["first", "grandchild", "sibling"]
    assert all(row["accepted"] for row in report["proposal_events"])


def test_queue_does_not_discard_an_unresolved_branch():
    root = root_state()
    missing = proposal("missing", "not-a-state")
    queue = FairProposalQueue([missing])
    assert queue.pop_ready({root.state_id: root}) is None
    assert [row.proposal_id for row in queue.unresolved()] == ["missing"]


def test_exact_closure_runs_shared_admission_but_never_emits_short_fixture():
    root = root_state()
    state, _ = apply_proposal(root, proposal("base", root.state_id, left="ab", right="ba"))
    assert state is not None
    final = Proposal("short-close", state.state_id, {"kind": "fixture"}, "finalize", center_text="")
    event = close_proposal(state, final)
    assert event["accepted"]
    assert event["letters"] == 4
    assert all("mechanical_checks" in row for row in event["closures"])
    assert candidate_records([event]) == []


def test_candidate_emission_predicate_requires_exact_100_plus_and_all_mechanical_checks():
    eligible = {"letters": 100, "exact_letter_palindrome": True, "mechanically_eligible": True}
    too_short = {"letters": 99, "exact_letter_palindrome": True, "mechanically_eligible": True}
    bad_gate = {"letters": 100, "exact_letter_palindrome": True, "mechanically_eligible": False}
    records = candidate_records([{"closures": [eligible, too_short, bad_gate]}])
    assert records == [eligible]


def test_cli_fixture_writes_full_ledger_and_leaves_reader_study_untriggered(tmp_path):
    output = tmp_path / "pilot.json"
    completed = subprocess.run(
        [sys.executable, "experiments/assisted_candidate_construction_pilot_20260913.py", "--output", str(output)],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads(completed.stdout)
    report = json.loads(output.read_text())
    assert summary["eligible_100_plus_candidates"] == 0
    assert report["config"]["model_calls"] == "none"
    assert report["proposal_events"]
    assert report["finalization_events"]
    assert report["human_reader_study"]["triggered"] is False


def test_pilot_does_not_contain_a_model_client_or_claim_readability():
    source = Path("experiments/assisted_candidate_construction_pilot_20260913.py").read_text().lower()
    assert "ollama" not in source
    assert "openai(" not in source
    assert "readability certification" in source
