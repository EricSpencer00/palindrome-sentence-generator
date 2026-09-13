"""Bounded local-proposer driver over the assisted candidate-construction kernel.

The driver predeclares twelve calls, each of which must return four strict JSON
alternatives.  It is intentionally thin: proposal schema validation, raw
prompt/reply provenance, and fair parent-state selection live here; every
character, debt, boundary, reopening, and admission decision is delegated to
``assisted_candidate_construction_pilot_20260913``.  It has no grammar and no
machine readability certification.

The default CLI is replay-only.  A real local request needs explicit ``--live``
so running tests or inspecting this module never contacts a model.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Protocol
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.assisted_candidate_construction_pilot_20260913 import (
    ConstructionState,
    Proposal,
    Region,
    apply_proposal,
    candidate_records,
    close_proposal,
    root_state,
    state_diagnostic,
)


DATE = "20260913"
PROPOSAL_CALLS = 12
ALTERNATIVES_PER_CALL = 4
DEFAULT_SEED = 20260913
HOST = "http://localhost:11434"


class LocalProposerClient(Protocol):
    """Minimal, mockable boundary around a local text proposer."""

    def metadata(self, model: str) -> dict[str, Any]: ...

    def complete(self, *, model: str, prompt: str, seed: int) -> str: ...


class OllamaLocalClient:
    """Explicitly opt-in local client; tests use a fake conforming client."""

    def __init__(self, host: str = HOST):
        self.host = host.rstrip("/")

    def _request(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(self.host + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)

    def metadata(self, model: str) -> dict[str, Any]:
        return self._request("/api/show", {"name": model})

    def complete(self, *, model: str, prompt: str, seed: int) -> str:
        response = self._request("/api/chat", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": "low",
            "options": {"temperature": 0.7, "seed": seed},
        })
        content = response.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("local_response_has_no_message_content")
        return content


class ReplayClient:
    """Filesystem-only client used for deterministic tests and dry replays."""

    def __init__(self, replies: list[str], metadata: dict[str, Any] | None = None):
        self.replies = list(replies)
        self._metadata = metadata or {"kind": "replay", "model": "mock"}
        self.calls: list[dict[str, Any]] = []

    def metadata(self, model: str) -> dict[str, Any]:
        return {**self._metadata, "requested_model": model}

    def complete(self, *, model: str, prompt: str, seed: int) -> str:
        self.calls.append({"model": model, "prompt": prompt, "seed": seed})
        if not self.replies:
            raise ValueError("replay_has_fewer_than_twelve_replies")
        return self.replies.pop(0)


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def visible_context(state: ConstructionState) -> dict[str, Any]:
    """Everything the proposer may see about its selected parent state."""
    diagnostic = state_diagnostic(state)
    return {
        "parent_state_id": state.state_id,
        "left_committed_visible_fringe": state.left_tape,
        "right_committed_visible_fringe": state.right_tape,
        "right_is_in_natural_reading_direction": True,
        "outstanding_symmetric_debt": diagnostic["symmetric_debt"],
        "left_boundary_analysis": diagnostic["left"],
        "right_boundary_analysis": diagnostic["right"],
        "full_edit_provenance": list(state.edit_history),
        "proposal_chain": list(state.proposal_history),
    }


def prompt_for(state: ConstructionState, *, call_index: int, seed: int) -> str:
    """Frozen strict-JSON request.  It asks for edits, never a full candidate."""
    schema = {
        "alternatives": [
            {
                "operation": "continue",
                "left_text": "lowercase ascii letters and spaces",
                "right_text": "lowercase ascii letters and spaces",
                "notes": "short repair intent",
            },
            {
                "operation": "reopen",
                "left_region": {"start": 0, "end": 0, "text": "replacement"},
                "right_region": {"start": 0, "end": 0, "text": "replacement"},
                "notes": "short repair intent",
            },
        ],
    }
    payload = {
        "task": "Propose four bounded coordinated edits to an exact-palindrome construction ledger.",
        "call_index": call_index,
        "seed": seed,
        "visible_parent_state": visible_context(state),
        "hard_rules": [
            "Return one JSON object and nothing else.",
            "Return exactly four alternatives.",
            "Each alternative is either continue or reopen; never finalize or a full sentence.",
            "For continue, supply nonempty left_text and right_text independently. Do not derive one from the other.",
            "For reopen, supply both replacement regions with normalized character offsets against the visible fringes.",
            "Each supplied text is at most 48 normalized letters. Do not use punctuation, commentary, a word-order mirror, or a catalogue palindrome.",
            "The host, not you, checks character symmetry, all lexical boundary analyses, and admission. Do not claim readability.",
        ],
        "response_schema_examples": schema,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _strict_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"reply_is_not_strict_json:{error.msg}") from error
    if not isinstance(value, dict) or set(value) != {"alternatives"}:
        raise ValueError("reply_must_have_only_alternatives_field")
    alternatives = value["alternatives"]
    if not isinstance(alternatives, list) or len(alternatives) != ALTERNATIVES_PER_CALL:
        raise ValueError("reply_must_have_exactly_four_alternatives")
    return value


def parse_alternatives(raw: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Validate every field before handing anything to the pilot kernel."""
    try:
        value = _strict_object(raw)
        alternatives: list[dict[str, Any]] = []
        fingerprints = set()
        for row in value["alternatives"]:
            if not isinstance(row, dict) or not isinstance(row.get("operation"), str):
                raise ValueError("alternative_requires_operation")
            operation = row["operation"]
            if operation == "continue":
                if set(row) != {"operation", "left_text", "right_text", "notes"}:
                    raise ValueError("continue_has_wrong_fields")
                if not all(isinstance(row[key], str) for key in ("left_text", "right_text", "notes")):
                    raise ValueError("continue_fields_must_be_strings")
                if not row["left_text"].strip() or not row["right_text"].strip():
                    raise ValueError("continue_requires_both_nonempty_sides")
            elif operation == "reopen":
                if set(row) != {"operation", "left_region", "right_region", "notes"}:
                    raise ValueError("reopen_has_wrong_fields")
                for key in ("left_region", "right_region"):
                    region = row[key]
                    if not isinstance(region, dict) or set(region) != {"start", "end", "text"}:
                        raise ValueError("reopen_region_has_wrong_fields")
                    if not isinstance(region["start"], int) or not isinstance(region["end"], int) or not isinstance(region["text"], str):
                        raise ValueError("reopen_region_types_invalid")
                if not isinstance(row["notes"], str):
                    raise ValueError("reopen_notes_must_be_string")
            else:
                raise ValueError("alternative_operation_must_be_continue_or_reopen")
            fingerprint = json.dumps(row, sort_keys=True, separators=(",", ":"))
            if fingerprint in fingerprints:
                raise ValueError("alternatives_must_be_distinct")
            fingerprints.add(fingerprint)
            alternatives.append(row)
        return alternatives, None
    except ValueError as error:
        return None, str(error)


def proposal_from_alternative(*, call_index: int, alternative_index: int, parent: ConstructionState,
                              alternative: dict[str, Any], model: str, seed: int,
                              reply_sha256: str) -> Proposal:
    source = {"kind": "model", "model": model, "seed": str(seed), "call_index": str(call_index),
              "alternative_index": str(alternative_index), "reply_sha256": reply_sha256}
    identifier = f"call-{call_index:02d}-alternative-{alternative_index:02d}"
    if alternative["operation"] == "continue":
        return Proposal(identifier, parent.state_id, source, "continue", left_text=alternative["left_text"],
                        right_text=alternative["right_text"], notes=alternative["notes"])
    return Proposal(identifier, parent.state_id, source, "reopen",
                    left_region=Region(**alternative["left_region"]), right_region=Region(**alternative["right_region"]),
                    notes=alternative["notes"])


def run_local_proposer(client: LocalProposerClient, *, model: str, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Perform exactly 12 calls × 4 alternatives through the pilot kernel."""
    model_metadata = client.metadata(model)
    root = root_state()
    states: dict[str, ConstructionState] = {root.state_id: root}
    order = [root.state_id]
    call_records, kernel_events, closure_events = [], [], []
    for call_index in range(PROPOSAL_CALLS):
        parent_id = order[call_index % len(order)]
        parent = states[parent_id]
        call_seed = seed + call_index
        prompt = prompt_for(parent, call_index=call_index, seed=call_seed)
        record = {"call_index": call_index, "parent_state_id": parent_id, "seed": call_seed,
                  "model": model, "model_metadata": model_metadata, "prompt": prompt,
                  "prompt_sha256": _digest(prompt)}
        try:
            raw_reply = client.complete(model=model, prompt=prompt, seed=call_seed)
            if not isinstance(raw_reply, str):
                raise ValueError("client_reply_not_string")
            record["raw_reply"] = raw_reply
            record["reply_sha256"] = _digest(raw_reply)
            alternatives, rejection = parse_alternatives(raw_reply)
            record["schema_rejection"] = rejection
            record["parsed_alternatives"] = alternatives
            if alternatives is None:
                call_records.append(record)
                continue
            per_call = []
            for alternative_index, alternative in enumerate(alternatives):
                proposal = proposal_from_alternative(call_index=call_index, alternative_index=alternative_index,
                                                     parent=parent, alternative=alternative, model=model, seed=call_seed,
                                                     reply_sha256=record["reply_sha256"])
                child, event = apply_proposal(parent, proposal)
                per_call.append(event)
                kernel_events.append(event)
                if child is not None:
                    states[child.state_id] = child
                    order.append(child.state_id)
                    # Empty-centre audit is a pure kernel call; it cannot add
                    # text or synthesize a counterpart, but it records any
                    # actual exact closure now available from this proposal.
                    audit = Proposal(f"{proposal.proposal_id}-empty-centre-audit", child.state_id,
                                     {"kind": "kernel_audit", "parent_model": model, "seed": str(call_seed)},
                                     "finalize", center_text="")
                    closure_events.append(close_proposal(child, audit))
            record["kernel_events"] = per_call
        except (OSError, ValueError, json.JSONDecodeError) as error:
            record["driver_rejection"] = str(error)
        call_records.append(record)
    candidates = candidate_records(closure_events)
    return {
        "status": "bounded_local_assisted_proposer_driver_complete",
        "config": {"date": DATE, "proposal_calls": PROPOSAL_CALLS, "alternatives_per_call": ALTERNATIVES_PER_CALL,
                   "seed": seed, "model": model, "new_grammar": False,
                   "kernel": "assisted_candidate_construction_pilot_20260913",
                   "counterpart_synthesis": "forbidden", "machine_readability_certification": False},
        "model_metadata": model_metadata, "root_state_id": root.state_id,
        "calls": call_records, "kernel_proposal_events": kernel_events, "empty_center_closure_audits": closure_events,
        "states_created": len(states), "eligible_100_plus_candidates": candidates,
        "human_reader_study": {"triggered": False,
                                "reason": "no mechanically eligible 100+ exact candidate" if not candidates else "candidate exists; no reader study has been run"},
        "source_provenance": {"driver_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                              "all_raw_prompts_and_replies_retained": True},
    }


def _replay_client(path: Path) -> ReplayClient:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or set(value) - {"metadata", "replies"}:
        raise ValueError("replay file must contain only metadata and replies")
    replies = value.get("replies")
    if not isinstance(replies, list) or not all(isinstance(row, str) for row in replies):
        raise ValueError("replay replies must be a list of JSON strings")
    if len(replies) != PROPOSAL_CALLS:
        raise ValueError("replay must provide exactly twelve replies")
    metadata = value.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("replay metadata must be an object")
    return ReplayClient(replies, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--replay", type=Path, help="deterministic 12-reply JSON ledger")
    source.add_argument("--live", action="store_true", help="explicitly permit 12 local requests")
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client: LocalProposerClient = _replay_client(args.replay) if args.replay else OllamaLocalClient(args.host)
    result = run_local_proposer(client, model=args.model, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "calls": len(result["calls"]),
                      "kernel_events": len(result["kernel_proposal_events"]),
                      "eligible_100_plus_candidates": len(result["eligible_100_plus_candidates"]),
                      "human_reader_study_triggered": result["human_reader_study"]["triggered"]}, sort_keys=True))


if __name__ == "__main__":
    main()
