import json
from pathlib import Path
import subprocess
import sys

from experiments.local_assisted_proposer_driver_20260913 import (
    ALTERNATIVES_PER_CALL,
    PROPOSAL_CALLS,
    ReplayClient,
    parse_alternatives,
    prompt_for,
    root_state,
    run_local_proposer,
)


def valid_reply(offset=0):
    alphabet = "abcdefghijklmnop"
    alternatives = []
    for index in range(ALTERNATIVES_PER_CALL):
        left = alphabet[offset + index:offset + index + 2]
        alternatives.append({"operation": "continue", "left_text": left, "right_text": left[::-1],
                             "notes": f"independent paired repair {index}"})
    return json.dumps({"alternatives": alternatives})


class MockLocalClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.metadata_calls = []
        self.complete_calls = []

    def metadata(self, model):
        self.metadata_calls.append(model)
        return {"model": model, "backend": "mock", "digest": "mock-digest"}

    def complete(self, *, model, prompt, seed):
        self.complete_calls.append({"model": model, "prompt": prompt, "seed": seed})
        return self.replies.pop(0)


def test_prompt_exposes_complete_fringe_debt_boundary_and_provenance_but_not_a_candidate_request():
    prompt = json.loads(prompt_for(root_state(), call_index=0, seed=13))
    visible = prompt["visible_parent_state"]
    assert {"left_committed_visible_fringe", "right_committed_visible_fringe", "outstanding_symmetric_debt",
            "left_boundary_analysis", "right_boundary_analysis", "full_edit_provenance"} <= set(visible)
    assert prompt["hard_rules"][3].startswith("For continue, supply nonempty left_text and right_text independently")
    assert all("readability" not in value.lower() or "do not claim" in value.lower() for value in prompt["hard_rules"])


def test_strict_schema_requires_exactly_four_distinct_coordinated_alternatives():
    parsed, rejection = parse_alternatives(valid_reply())
    assert rejection is None and len(parsed) == 4
    invalid, rejection = parse_alternatives(json.dumps({"alternatives": [{"operation": "continue"}]}))
    assert invalid is None
    assert rejection == "reply_must_have_exactly_four_alternatives"
    duplicate = json.loads(valid_reply())
    duplicate["alternatives"][1] = duplicate["alternatives"][0]
    invalid, rejection = parse_alternatives(json.dumps(duplicate))
    assert invalid is None and rejection == "alternatives_must_be_distinct"


def test_mocked_driver_makes_predeclared_12_by_4_calls_and_preserves_all_raw_artifacts():
    client = MockLocalClient([valid_reply() for _ in range(PROPOSAL_CALLS)])
    report = run_local_proposer(client, model="mock-local", seed=700)
    assert len(client.complete_calls) == PROPOSAL_CALLS
    assert len(report["calls"]) == PROPOSAL_CALLS
    assert len(report["kernel_proposal_events"]) == PROPOSAL_CALLS * ALTERNATIVES_PER_CALL
    assert report["config"]["proposal_calls"] == 12
    assert report["config"]["alternatives_per_call"] == 4
    assert report["config"]["new_grammar"] is False
    assert report["config"]["counterpart_synthesis"] == "forbidden"
    for index, record in enumerate(report["calls"]):
        assert record["seed"] == 700 + index
        assert record["raw_reply"]
        assert record["prompt_sha256"]
        assert record["model_metadata"]["backend"] == "mock"
        assert len(record["kernel_events"]) == 4
    assert report["human_reader_study"]["triggered"] is False


def test_schema_rejections_are_logged_for_every_failed_reply_without_silently_dropping_calls():
    client = MockLocalClient(["not json" for _ in range(PROPOSAL_CALLS)])
    report = run_local_proposer(client, model="mock-local")
    assert len(client.complete_calls) == PROPOSAL_CALLS
    assert len(report["calls"]) == PROPOSAL_CALLS
    assert all(record["schema_rejection"].startswith("reply_is_not_strict_json") for record in report["calls"])
    assert report["kernel_proposal_events"] == []


def test_replay_client_and_cli_are_filesystem_only_by_default(tmp_path):
    replay = tmp_path / "replay.json"
    output = tmp_path / "driver.json"
    replay.write_text(json.dumps({"metadata": {"backend": "replay"}, "replies": [valid_reply() for _ in range(12)]}))
    completed = subprocess.run(
        [sys.executable, "experiments/local_assisted_proposer_driver_20260913.py", "--replay", str(replay),
         "--model", "replay-model", "--output", str(output)],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads(completed.stdout)
    report = json.loads(output.read_text())
    assert summary["calls"] == 12
    assert report["model_metadata"]["backend"] == "replay"
    assert report["human_reader_study"]["triggered"] is False


def test_driver_source_does_not_import_or_create_another_grammar():
    source = Path("experiments/local_assisted_proposer_driver_20260913.py").read_text()
    assert "from experiments.assisted_candidate_construction_pilot_20260913 import" in source
    assert "compile_slots" not in source
    assert "mechanical_admission_checks" not in source
