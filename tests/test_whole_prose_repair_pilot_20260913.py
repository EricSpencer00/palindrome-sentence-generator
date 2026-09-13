import json
from pathlib import Path
import subprocess
import sys

from experiments.whole_prose_repair_pilot_20260913 import (
    ALTERNATIVES_PER_REPAIR,
    LINEAGES,
    REPAIR_ROUNDS,
    independent_ascii_letters,
    independent_exactness,
    initialization_prompt,
    parse_initialization,
    parse_repairs,
    repair_prompt,
    run_whole_prose_repair_pilot,
    symmetry_diagnostics,
)


def initial_reply():
    drafts = []
    for index in range(LINEAGES):
        text = "The patient curator carefully reviewed every label before closing the quiet exhibit, because visitors needed a clear explanation."
        drafts.append({"intent": f"ordinary scene {index}", "text": text})
    return json.dumps({"drafts": drafts})


def repair_reply(round_index=0):
    alternatives = []
    for index in range(ALTERNATIVES_PER_REPAIR):
        text = "The patient curator carefully reviewed every label before closing the quiet exhibit, because visitors needed a clear explanation."
        alternatives.append({"text": text + ("" if index == 0 else " " + "a" * index), "notes": f"complete global revision {round_index}-{index}"})
    return json.dumps({"alternatives": alternatives})


def critique_reply(round_index=0):
    return json.dumps({"items": [{"lineage": index, "round": round_index, "defects": []} for index in range(LINEAGES)]})


class MockClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def metadata(self, model):
        return {"backend": "mock", "model": model, "digest": "mock-digest"}

    def complete(self, *, model, prompt, seed):
        self.calls.append({"model": model, "prompt": prompt, "seed": seed})
        return self.replies.pop(0)


def test_independent_normalization_and_mismatch_diagnostics_are_position_based():
    assert independent_ascii_letters("A dog, a panic in a pagoda!") == "adogapanicinapagoda"
    exact = independent_exactness("A dog, a panic in a pagoda!")
    assert exact["direct_symmetric_position_comparison"]
    diagnostics = symmetry_diagnostics("The dog watched the moon.")
    assert diagnostics["letters"] == len(diagnostics["normalized_tape"])
    assert diagnostics["mismatch_count"] > 0
    assert all({"left_index", "right_index", "left_word", "right_word"} <= set(pair) for pair in diagnostics["mismatches"])


def test_prompts_keep_complete_prose_and_do_not_request_mirrored_halves():
    initial = json.loads(initialization_prompt())
    assert "palindrome" not in json.dumps(initial).lower()
    rows, rejection = parse_initialization(initial_reply())
    assert rejection is None and len(rows) == LINEAGES
    from experiments.whole_prose_repair_pilot_20260913 import Draft

    prompt = json.loads(repair_prompt(Draft(0, 0, rows[0]["text"], rows[0]["intent"], None, {})))
    assert prompt["current_complete_prose"] == rows[0]["text"]
    assert "mismatches" in prompt["character_diagnostics"]
    assert all("independent halves" not in rule for rule in prompt["hard_rules"])
    assert "reverse" not in json.dumps(prompt["response_schema"]).lower()


def test_strict_repair_schema_requires_four_distinct_full_surfaces():
    rows, rejection = parse_repairs(repair_reply())
    assert rejection is None and len(rows) == ALTERNATIVES_PER_REPAIR
    bad = json.loads(repair_reply())
    bad["alternatives"][1] = bad["alternatives"][0]
    rows, rejection = parse_repairs(json.dumps(bad))
    assert rows is None and rejection == "repair_alternatives_must_be_distinct"


def test_bounded_driver_preserves_every_complete_surface_and_never_auto_claims_readability():
    replies = [initial_reply()]
    for round_index in range(REPAIR_ROUNDS):
        replies.extend(repair_reply(round_index) for _ in range(LINEAGES))
        replies.append(critique_reply(round_index))
    client = MockClient(replies)
    report = run_whole_prose_repair_pilot(client, model="mock", seed=900)
    assert len(client.calls) == 1 + LINEAGES * REPAIR_ROUNDS + REPAIR_ROUNDS
    assert len(report["repair_calls"]) == LINEAGES * REPAIR_ROUNDS
    assert len(report["critique_calls"]) == REPAIR_ROUNDS
    assert len(report["candidate_audits"]) == LINEAGES * REPAIR_ROUNDS
    assert report["config"]["counterpart_synthesis"] == "forbidden"
    assert report["config"]["construction_representation"] == "complete_prose_with_temporary_symmetry_errors"
    assert report["human_reader_study"]["triggered"] is False
    assert all(call["parent"]["text"] for call in report["repair_calls"])
    assert all(len(call["alternatives"]) == ALTERNATIVES_PER_REPAIR for call in report["repair_calls"])
    assert all(call["parsed_items"] is not None for call in report["critique_calls"])
    assert all(audit["human_reader_study"] == "not_run" for audit in report["candidate_audits"])


def test_replay_cli_is_filesystem_only(tmp_path):
    replay = tmp_path / "replay.json"
    output = tmp_path / "report.json"
    replies = [initial_reply()]
    for round_index in range(REPAIR_ROUNDS):
        replies.extend(repair_reply(round_index) for _ in range(LINEAGES))
        replies.append(critique_reply(round_index))
    replay.write_text(json.dumps({"metadata": {"backend": "replay"}, "replies": replies}))
    completed = subprocess.run(
        [sys.executable, "experiments/whole_prose_repair_pilot_20260913.py", "--replay", str(replay), "--output", str(output)],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads(completed.stdout)
    report = json.loads(output.read_text())
    assert summary["repair_calls"] == LINEAGES * REPAIR_ROUNDS
    assert report["model_metadata"]["backend"] == "replay"


def test_source_is_not_a_fixed_sentence_grammar_or_surface_synthesizer():
    source = Path("experiments/whole_prose_repair_pilot_20260913.py").read_text()
    assert "compile_slots" not in source
    assert "left_text" not in source
    assert "right_text" not in source
    assert "[::-1]" not in source
