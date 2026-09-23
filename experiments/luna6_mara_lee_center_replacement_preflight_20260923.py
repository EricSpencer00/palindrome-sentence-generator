"""Reproduce a bounded authored-prose replacement and its first obstruction."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "runs/luna6-god-dog-live-residual-growth-20260923.json"
OUTPUT_PATH = ROOT / "runs/luna6-mara-lee-center-replacement-preflight-20260923.json"
PARENT_SHA256 = "017d5e73f11339204b3f343fa66e63b5c4674afd01c5ea823e36999879a2ac11"
PREFLIGHT_REVISION = "bad78bbd"
START, END = 284, 346
PROPOSAL = (
    "Mara sent Lee one chart at dusk; later he quietly filed it in the main archive room."
)


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def positions(text: str) -> list[int]:
    return [i for i, c in enumerate(text) if c.isascii() and c.isalpha()]


def scan(tape: str) -> dict[str, object]:
    left, right = 0, len(tape) - 1
    while left < right and tape[left] == tape[right]:
        left += 1
        right -= 1
    exact = left >= right
    return {
        "exact": exact,
        "letters": len(tape),
        "matched_outer_pairs": left,
        "first_mismatch": None if exact else {
            "left_offset": left,
            "right_offset": right,
            "left": tape[left],
            "right": tape[right],
        },
    }


def main() -> dict[str, object]:
    sys.path.insert(0, str(ROOT))
    from llm_palindrome.validator import is_palindrome

    parent_record = json.loads(PARENT_PATH.read_text())
    parent = parent_record["rendered_full_text"]
    parent_tape = letters(parent)
    if len(parent_tape) != 630 or hashlib.sha256(parent_tape.encode()).hexdigest() != PARENT_SHA256:
        raise AssertionError("pinned 630 parent changed")
    if not scan(parent_tape)["exact"] or not is_palindrome(parent):
        raise AssertionError("pinned 630 parent is not exact")
    if not (0 <= START < END <= len(parent_tape)):
        raise AssertionError("replacement cursors outside parent tape")

    raw = positions(parent)
    source_surface = parent[raw[START]:raw[END - 1] + 1]
    source_tape = parent_tape[START:END]
    proposal_tape = letters(PROPOSAL)
    candidate = parent[:raw[START]] + PROPOSAL + parent[raw[END - 1] + 1:]
    candidate_tape = letters(candidate)
    result = scan(candidate_tape)
    project_exact = bool(is_palindrome(candidate))
    proposal_sha = hashlib.sha256(proposal_tape.encode()).hexdigest()
    candidate_sha = hashlib.sha256(candidate_tape.encode()).hexdigest()

    grep = subprocess.run(
        ["git", "grep", "-F", PROPOSAL, PREFLIGHT_REVISION, "--",
         "experiments", "runs", "docs", "data"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if grep.returncode not in (0, 1):
        raise RuntimeError(grep.stderr)
    phrase_absent = grep.returncode == 1
    if len(source_tape) != 62 or len(proposal_tape) != 66:
        raise AssertionError("unexpected source/proposal size")
    if result["exact"] or project_exact or result["first_mismatch"] != {
        "left_offset": 285,
        "right_offset": 348,
        "left": "a",
        "right": "o",
    }:
        raise AssertionError("bounded obstruction changed; inspect before reusing")

    return {
        "experiment_id": "luna6-mara-lee-center-replacement-preflight-20260923",
        "status": "not_exact_replacement_rejected_at_first_character_obstruction",
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "letters": len(parent_tape),
            "sha256": PARENT_SHA256,
            "exact": True,
        },
        "authored_proposal": {
            "rendered": PROPOSAL,
            "normalized": proposal_tape,
            "letters": len(proposal_tape),
            "sha256": proposal_sha,
            "complete_event_reading": "Mara sends Lee a chart; later he files it in their archive room.",
            "novelty_preflight": {
                "revision": PREFLIGHT_REVISION,
                "exact_literal_absent": phrase_absent,
                "broader_chart-handoff_family_already_seen": True,
            },
        },
        "replacement": {
            "normalized_cursors": [START, END],
            "source_letters": len(source_tape),
            "source_tape": source_tape,
            "source_surface": source_surface,
            "candidate_letters": len(candidate_tape),
            "first_obstruction": result["first_mismatch"],
            "outside_in": result,
            "project_validator_exact": project_exact,
            "candidate_sha256": candidate_sha,
            "full_candidate_not_admitted": True,
        },
        "repair_operator": {
            "rejected": "replace a complete exact interior span with an independently authored clause pair",
            "obstruction": "first two tape letters are M/A; the mirrored ending contributes R/O, yielding a/o mismatch at cursor 1",
            "next": "change to the distinct 305/325 live partial-word seam and grow through a paired event that propagates its residual; do not try another whole-span replacement",
        },
    }


if __name__ == "__main__":
    record = main()
    OUTPUT_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": record["status"],
        "proposal_letters": record["authored_proposal"]["letters"],
        "candidate_letters": record["replacement"]["candidate_letters"],
        "first_obstruction": record["replacement"]["first_obstruction"],
        "novelty_preflight_absent": record["authored_proposal"]["novelty_preflight"]["exact_literal_absent"],
    }, sort_keys=True))
