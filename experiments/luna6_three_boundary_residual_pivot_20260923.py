"""Persist one bounded residual probe and its changed whole-sentence seam."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT_PATH = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUTPUT_PATH = ROOT / "runs/luna6-three-boundary-residual-pivot-20260923.json"
EXPECTED_PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"


def letters(value: str) -> str:
    return "".join(c.lower() for c in value if c.isalpha())


def outside_in(value: str) -> dict:
    tape = letters(value)
    pairs = []
    first_mismatch = None
    for i in range(len(tape) // 2):
        j = len(tape) - i - 1
        equal = tape[i] == tape[j]
        pairs.append({"left": i, "right": j, "left_char": tape[i], "right_char": tape[j], "equal": equal})
        if not equal and first_mismatch is None:
            first_mismatch = {"left": i, "right": j, "left_char": tape[i], "right_char": tape[j]}
    return {"letters": len(tape), "exact": first_mismatch is None, "first_mismatch": first_mismatch, "pairs_checked": len(pairs)}


def main() -> None:
    from llm_palindrome.validator import is_palindrome as project_is_palindrome

    parent_json = json.loads(PARENT_PATH.read_text())
    parent_row = next(row for row in parent_json["rows"] if row["id"] == "outer-causal-scene-568-working-incumbent")
    parent_rendered = parent_row["rendered"]
    parent_tape = letters(parent_rendered)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    assert len(parent_tape) == 568
    assert parent_sha == EXPECTED_PARENT_SHA256
    assert outside_in(parent_rendered)["exact"]
    assert project_is_palindrome(parent_rendered)

    # One bounded human-authored lexicalization probe.  The 13-letter local
    # equation closes, but does so through the forbidden whole-token pair
    # said/Dias, so it is evidence, not an eligible scene or candidate.
    left_probe = "Nora, as I said"
    right_suffix_probe = "Dias is Aaron"
    left_tape = letters(left_probe)
    right_tape = letters(right_suffix_probe)
    required_left = right_tape[::-1]
    cursor = 0
    while cursor < min(len(left_tape), len(required_left)) and left_tape[cursor] == required_left[cursor]:
        cursor += 1
    assert left_tape == required_left
    boundary_audit = {
        "left_tokens": ["Nora", "as", "I", "said"],
        "right_tokens": ["Dias", "is", "Aaron"],
        "whole_token_reversal_pairs": [["said", "Dias"]],
        "self_palindromic_tokens": ["I"],
        "admissible": False,
        "reason": "The exact local closure uses the whole-token reversal said/Dias and the standalone self-palindromic token I; both are excluded.",
    }

    # The next seam starts at a real sentence boundary in the parent and is
    # distinct from the failed [135,232)+cut-433 ownership.
    next_start, next_end, next_cut = 91, 232, 477
    assert next_cut == len(parent_tape) - next_start
    next_span = parent_tape[next_start:next_end]
    assert len(next_span) == next_end - next_start == 141
    next_seam = {
        "replacement_parent_span": [next_start, next_end],
        "replacement_parent_letters": len(next_span),
        "opposing_insert_parent_cut": next_cut,
        "reflected_start": len(parent_tape) - next_cut,
        "parent_removed_tape": next_span,
        "rendered_flank_context": {
            "left": parent_rendered[: next((i for i, c in enumerate(parent_rendered) if sum(ch.isalpha() for ch in parent_rendered[:i]) >= next_start), 0)][-80:],
            "right": parent_tape[next_end : next_end + 32],
        },
        "boundary_note": "Normalized offset 91 begins `Aidan delivers maps`; offset 232 is the sentence boundary after `desserts`.",
    }

    result = {
        "experiment_id": "luna6-three-boundary-residual-pivot-20260923",
        "status": "no_admissible_70_letter_scene_pair; exact local shortcut rejected; changed seam selected",
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "letters": len(parent_tape),
            "sha256_normalized": parent_sha,
            "independent_outside_in_exact": outside_in(parent_rendered)["exact"],
            "project_validator_exact": project_is_palindrome(parent_rendered),
        },
        "preflight": {
            "current_geometry": {"replace_parent_span": [135, 232], "opposing_insert_parent_cut": 433},
            "registry_history": "No completed run uses this exact signature. It was previously recorded only as a next operator and appears in an unexecuted draft; this one bounded lexicalization pass found no admissible 70-letter pair.",
            "changed_geometry": next_seam,
            "changed_geometry_signature_found": False,
        },
        "bounded_local_attempt": {
            "left_rendered_probe": left_probe,
            "right_rendered_suffix": right_suffix_probe,
            "left_normalized": left_tape,
            "right_normalized": right_tape,
            "required_left_from_right": required_left,
            "letters_matched": cursor,
            "target_letters": 70,
            "full_70_letter_equation_closed": False,
            "residual": {
                "closed_prefix": left_tape,
                "remaining_positions": [cursor, 70],
                "remaining_count": 70 - cursor,
                "obstruction": "This authored continuation reaches only 13 characters and closes only by the forbidden said/Dias whole-token reversal. No coherent admissible 70-letter pair was supplied, so no full candidate was rendered.",
            },
            "boundary_audit": boundary_audit,
        },
        "candidate": None,
        "next_operator": "Preflight and author once on sentence-bounded [91,232) with a new event inserted at cut 477; carry the resulting 70-character equation while composing, and reject any closure that uses tokenwise reversal.",
        "reader_status": "not_applicable_no_candidate; no readability claim",
        "provenance": {
            "lexical_source": "one fresh hand-authored diagnostic phrase pair; not copied into a candidate",
            "whole_tape_rendered": False,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT_PATH), "experiment_id": result["experiment_id"], "status": result["status"], "matched": f"{cursor}/70", "next_seam": next_seam["replacement_parent_span"] + [next_cut]}, indent=2))


if __name__ == "__main__":
    main()
