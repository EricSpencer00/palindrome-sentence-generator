#!/usr/bin/env python3
"""Reconstruct the outer endpoint-scene attempt and its exact obstruction."""
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
OUT_PATH = ROOT / "runs" / "audit-incumbent-568-outer-endpoint-residual-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
PREFLIGHT_REVISION = "2cd8a393"
LEFT_SPAN = (0, 148)
RIGHT_SPAN = (420, 568)
LEFT_RAW = (0, 195)
RIGHT_RAW = (582, 786)
OLD_LEFT = "Leon won. Wolf spots Nora. Nadia stops, so Tara rewards Nadia. Nora delivers maps. Mara stops rats. A tub? He maps Aron. Aidan delivers maps. Mara stops rats. A tub? He maps Nora. Deliver no evil"
OLD_RIGHT = "Live on, reviled. Aron, spam. Eh, but a star spots Aram. Spam's reviled, Nadia. Nora, spam. Eh, but a star spots Aram. Spam's reviled, Aron. Aidan's drawer, Aratos, spots Aidan. Aron stops flow now, Noel."
NEW_LEFT = "Noting frost, the surveyor secured the workshop. The custodian moved the packet to the inner room; checked each clasp, listed the damage, and sent the notice for review before dawn to headquarters today without further delay"
NEW_RIGHT = "After the breach was found, one steward inspected the hinge. Because the route was blocked, one driver redirected the crates, while one witness logged each change and the night crew confirmed delivery. The crew carried one ton"
EXPECTED_CHILD_SHA256 = "ad68f263f7d1ee681ef60a5d1cbcf80734fb0fbfbabf3f7a96b2d825ae40fb4c"


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
        raise AssertionError("outer source fragments differ from the reported parent")
    if parent_tape[LEFT_SPAN[0] : LEFT_SPAN[1]] != normalize(OLD_LEFT):
        raise AssertionError("left normalized span mismatch")
    if parent_tape[RIGHT_SPAN[0] : RIGHT_SPAN[1]] != normalize(OLD_RIGHT):
        raise AssertionError("right normalized span mismatch")
    if parent_tape[LEFT_SPAN[0] : LEFT_SPAN[1]] != parent_tape[RIGHT_SPAN[0] : RIGHT_SPAN[1]][::-1]:
        raise AssertionError("outer source fragments are not reflected")

    rendered = NEW_LEFT + parent[LEFT_RAW[1] : RIGHT_RAW[0]] + NEW_RIGHT
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
        raise AssertionError("reconstructed child differs from the reported candidate")
    if scan["exact"] or project_exact or forward == reverse:
        raise AssertionError("reported candidate unexpectedly closes")

    parallel_preflight = phrase_preflight((
        "noting frost",
        "surveyor secured the workshop",
        "custodian moved the packet",
        "one steward inspected the hinge",
        "one driver redirected the crates",
        "one witness logged each change",
    ))
    return {
        "experiment_id": "audit-incumbent-568-outer-endpoint-residual-20260923",
        "working_status": "rejected_nonexact_unequal_insert_lengths",
        "parent": {"artifact": str(PARENT_PATH.relative_to(ROOT)), "letters": len(parent_tape), "sha256_normalized": parent_sha},
        "operator": {
            "name": "endpoint-signature-first bidirectional causal-scene draft",
            "provenance": "read-only Luna matrix-scene lane; complete child independently reconstructed here",
            "preflight_revision": PREFLIGHT_REVISION,
            "normalized_spans": [list(LEFT_SPAN), list(RIGHT_SPAN)],
            "raw_spans": [list(LEFT_RAW), list(RIGHT_RAW)],
            "source_blocks": [OLD_LEFT, OLD_RIGHT],
            "insertions": [NEW_LEFT, NEW_RIGHT],
            "endpoint_signature": "left `not` equals the first three letters of reverse(right `one ton`)",
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
        "novelty_preflight": parallel_preflight,
        "admission": {
            "admitted": False,
            "reason": "The local tape matches `not` then diverges i/e; insertions also have unequal lengths (184/185), so this reflected seam cannot close, despite +73 letters. The left semicolon clause drops its subject and the right repeats `one`/`crew` scaffolding.",
            "human_readability_certified": False,
            "reader_evidence": False,
        },
        "next_construction": "On a different actual seam, require equal insertion lengths and a target strictly above the removed span as construction invariants before expanding either clause; retain the endpoint residual online.",
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
