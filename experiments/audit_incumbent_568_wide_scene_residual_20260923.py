#!/usr/bin/env python3
"""Reconstruct a wide causal-scene replacement and its live residual."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.validator import is_palindrome as project_is_palindrome

PARENT_PATH = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT_PATH = ROOT / "runs" / "audit-incumbent-568-wide-scene-residual-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
PREFLIGHT_REVISION = "c07a68cc"
LEFT_SPAN = (20, 148)
RIGHT_SPAN = (420, 548)
LEFT_RAW = (27, 195)
RIGHT_RAW = (582, 758)
OLD_LEFT = "Nadia stops, so Tara rewards Nadia. Nora delivers maps. Mara stops rats. A tub? He maps Aron. Aidan delivers maps. Mara stops rats. A tub? He maps Nora. Deliver no evil"
OLD_RIGHT = "Live on, reviled. Aron, spam. Eh, but a star spots Aram. Spam's reviled, Nadia. Nora, spam. Eh, but a star spots Aram. Spam's reviled, Aron. Aidan's drawer, Aratos, spots Aidan"
NEW_LEFT = "Noticing rain, the courier secured the observatory. The latch delayed the register, so the keeper moved records indoors. Damp notes preserved proof in full"
NEW_RIGHT = "After warning, the steward examined the breach. One broken hinge delayed delivery, so the clerk rerouted sealed files, so the team at sunrise carried one ton"
EXPECTED_CHILD_SHA256 = "88f122fa227833415684eebc68f31cf25edac55a316a86ad747a97fd54669c4e"


def normalize(text: str) -> str:
    return "".join(char.lower() for char in text if char.isascii() and char.isalpha())


def outside_in(tape: str) -> dict[str, Any]:
    left, right = 0, len(tape) - 1
    comparisons = 0
    while left < right:
        comparisons += 1
        if tape[left] != tape[right]:
            return {"exact": False, "first_mismatch": [left, tape[left], right, tape[right]], "comparisons": comparisons}
        left += 1
        right -= 1
    return {"exact": True, "first_mismatch": None, "comparisons": comparisons}


def phrase_preflight(phrases: tuple[str, ...]) -> dict[str, Any]:
    command = ["git", "grep", "-n", "-F", "-i"]
    for phrase in phrases:
        command.extend(("-e", phrase))
    command.extend((PREFLIGHT_REVISION, "--", "runs", "experiments", "docs", "data"))
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"phrase preflight failed: {result.stderr}")
    hits = result.stdout.splitlines()
    return {
        "revision": PREFLIGHT_REVISION,
        "phrases": list(phrases),
        "scope": "tracked runs/, experiments/, docs/, and data/",
        "hits": hits,
        "status": "no_literal_hits" if not hits else "phrase_family_collision",
    }


def build_payload() -> dict[str, Any]:
    source = json.loads(PARENT_PATH.read_text())
    row = next(item for item in source["rows"] if item["working_status"] == "working_length_incumbent")
    parent = str(row["rendered"])
    parent_tape = normalize(parent)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if len(parent_tape) != 568 or parent_sha != PARENT_SHA256:
        raise AssertionError("pinned 568 parent changed")
    if parent[LEFT_RAW[0] : LEFT_RAW[1]] != OLD_LEFT or parent[RIGHT_RAW[0] : RIGHT_RAW[1]] != OLD_RIGHT:
        raise AssertionError("reported raw scene spans differ from the parent")
    if parent_tape[LEFT_SPAN[0] : LEFT_SPAN[1]] != normalize(OLD_LEFT):
        raise AssertionError("left normalized scene span mismatch")
    if parent_tape[RIGHT_SPAN[0] : RIGHT_SPAN[1]] != normalize(OLD_RIGHT):
        raise AssertionError("right normalized scene span mismatch")
    if parent_tape[LEFT_SPAN[0] : LEFT_SPAN[1]] != parent_tape[RIGHT_SPAN[0] : RIGHT_SPAN[1]][::-1]:
        raise AssertionError("parent scene spans are not reflected")

    rendered = parent[: LEFT_RAW[0]] + NEW_LEFT + parent[LEFT_RAW[1] : RIGHT_RAW[0]] + NEW_RIGHT + parent[RIGHT_RAW[1] :]
    tape = normalize(rendered)
    left_tape, right_tape = normalize(NEW_LEFT), normalize(NEW_RIGHT)
    required = right_tape[::-1]
    cursor = 0
    while cursor < min(len(left_tape), len(required)) and left_tape[cursor] == required[cursor]:
        cursor += 1
    scan = outside_in(tape)
    forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    project_exact = project_is_palindrome(rendered)
    if forward != EXPECTED_CHILD_SHA256:
        raise AssertionError("reconstructed child differs from the reported hash")
    if scan["exact"] or project_exact or forward == reverse:
        raise AssertionError("reported scene unexpectedly closes")
    if len(left_tape) != len(right_tape):
        raise AssertionError("reported insertions should be equal length")

    testimony_tapes = [parent_tape[148:204], parent_tape[364:420]]
    if testimony_tapes[0] != testimony_tapes[1][::-1]:
        raise AssertionError("parallel testimony report geometry is invalid")
    preflight = phrase_preflight((
        "noticing rain",
        "courier secured the observatory",
        "latch delayed the register",
        "steward examined the breach",
        "broken hinge delayed delivery",
        "clerk rerouted sealed files",
    ))
    return {
        "experiment_id": "audit-incumbent-568-wide-scene-residual-20260923",
        "working_status": "rejected_nonexact_no_length_growth",
        "parent": {"artifact": str(PARENT_PATH.relative_to(ROOT)), "letters": len(parent_tape), "sha256_normalized": parent_sha},
        "operator": {
            "name": "context-clean wide causal scene replacement with live residual",
            "provenance": "read-only Luna matrix-scene lane; complete child reconstructed independently here",
            "preflight_revision": PREFLIGHT_REVISION,
            "normalized_spans": [list(LEFT_SPAN), list(RIGHT_SPAN)],
            "raw_spans": [list(LEFT_RAW), list(RIGHT_RAW)],
            "source_blocks": [OLD_LEFT, OLD_RIGHT],
            "insertions": [NEW_LEFT, NEW_RIGHT],
            "retained_context": ["Wolf spots Nora. / Now, Noel, did I live?", "... Leon won. / Aron stops flow now, Noel."],
        },
        "candidate": {
            "rendered": rendered,
            "letters": len(tape),
            "growth_over_parent": len(tape) - len(parent_tape),
            "normalized_sha256": forward,
            "audit": {
                "independent_outside_in_exact": scan["exact"],
                "first_mismatch": scan["first_mismatch"],
                "project_validator_exact": project_exact,
                "sha256_forward": forward,
                "sha256_reverse": reverse,
                "hashes_equal": forward == reverse,
            },
            "local_equation": {
                "left_tape": left_tape,
                "right_tape": right_tape,
                "required_reverse_right_tape": required,
                "left_letters": len(left_tape),
                "right_letters": len(right_tape),
                "matched_prefix_letters": cursor,
                "first_mismatch": {"cursor": cursor, "left": left_tape[cursor], "required": required[cursor]},
            },
        },
        "novelty_preflight": preflight,
        "parallel_lane_obstructions": [
            {
                "lane": "testimony-evidence-dialogue",
                "normalized_spans": [[148, 204], [364, 420]],
                "source_tapes": testimony_tapes,
                "fresh_units": 384,
                "complete_dialogue_states": 18816,
                "long_attempts": 7416,
                "exact_closures": 0,
                "furthest_prefix_letters": 1,
                "reported_frontier": "Rina reports that Mara heard Mara. Mara denies that Mara found letter.",
                "reported_residual": "inareportsthatmaraheardmaramaradeniesthatmarafoundletter",
                "candidate_hash": None,
                "provenance": "read-only Luna report; reflected geometry independently checked here; dialogue chart not rerun",
            }
        ],
        "admission": {
            "admitted": False,
            "reason": "The 128-letter insertions only replace 128-letter source blocks (no length growth), and the local equation matches `not` before diverging i/e at cursor 3; whole-text mismatch is at offset 23.",
            "human_readability_certified": False,
            "reader_evidence": False,
        },
        "next_construction": "Change the actual seam and require a target insertion length strictly greater than the removed block before prose expansion; make reverse-tape continuation a live stopping condition, not a post-hoc check.",
    }


if __name__ == "__main__":
    payload = build_payload()
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    candidate = payload["candidate"]
    print(json.dumps({
        "status": payload["working_status"],
        "letters": candidate["letters"],
        "sha256": candidate["normalized_sha256"],
        "insert_lengths": [candidate["local_equation"]["left_letters"], candidate["local_equation"]["right_letters"]],
        "cursor": candidate["local_equation"]["matched_prefix_letters"],
        "phrase_preflight": payload["novelty_preflight"]["status"],
    }, indent=2))
