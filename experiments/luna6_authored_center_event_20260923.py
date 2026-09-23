"""Audit one fresh authored event-span candidate at the pinned 568 midpoint."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUTPUT = ROOT / "runs/luna6-authored-center-event-20260923.json"
PARENT_SHA = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
CANDIDATE = "No, Mar, he saw Eve; Eve, was he Ramon?"


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def scan(tape: str) -> dict:
    i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]:
        i += 1
        j -= 1
    return {
        "exact": i >= j,
        "letters": len(tape),
        "matched_outer_pairs": i,
        "first_mismatch": None if i >= j else {
            "offset_left": i, "offset_right": j,
            "left": tape[i], "right": tape[j],
        },
        "left_residual": tape[i:i + 40],
        "right_reverse_residual": tape[max(0, j - 39):j + 1][::-1],
    }


def raw_offset(text: str, normalized_offset: int) -> int:
    count = 0
    for index, char in enumerate(text):
        if char.isascii() and char.isalpha():
            if count == normalized_offset:
                return index
            count += 1
    if count == normalized_offset:
        return len(text)
    raise ValueError(normalized_offset)


def main() -> dict:
    sys.path.insert(0, str(ROOT))
    from llm_palindrome.validator import is_palindrome

    source = json.loads(PARENT.read_text())
    parent = next(row["rendered"] for row in source["rows"]
                  if row["id"] == "outer-causal-scene-568-working-incumbent")
    parent_tape = letters(parent)
    if len(parent_tape) != 568 or sha(parent_tape) != PARENT_SHA:
        raise AssertionError("pinned parent identity changed")
    if not scan(parent_tape)["exact"] or not is_palindrome(parent):
        raise AssertionError("pinned parent failed exact validation")

    local = letters(CANDIDATE)
    local_scan = scan(local)
    center = 284
    raw = raw_offset(parent, center)
    attempted = parent[:raw] + CANDIDATE + " " + parent[raw:]
    full_tape = letters(attempted)
    full_scan = scan(full_tape)
    project_exact = bool(is_palindrome(attempted))
    fwd, rev = sha(full_tape), sha(full_tape[::-1])
    tokens = [token.lower() for token in CANDIDATE.replace(";", "").replace(",", "").replace("?", "").split()]
    token_reversals = sorted({(a, b) for a in tokens for b in tokens
                              if a != b and a[::-1] == b})
    self_pal = sorted({token for token in tokens if token == token[::-1]})
    singleton = sorted({token for token in tokens if len(token) == 1})
    repeated = sorted({token for token in set(tokens) if tokens.count(token) > 1})

    return {
        "experiment_id": "luna6-authored-center-event-20260923",
        "status": "authored_center_span_fails_exactness",
        "pinned_parent": {"letters": len(parent_tape), "sha256": PARENT_SHA,
                          "exact_outside_in": scan(parent_tape)["exact"],
                          "project_validator_exact": bool(is_palindrome(parent))},
        "construction": {
            "method": "one authored event-like span inserted at the normalized midpoint",
            "center_cut": center,
            "literal": CANDIDATE,
            "provenance": "freshly composed for this attempt; not claimed as catalogue text",
            "local_normalized_tape": local,
            "local_reverse": local[::-1],
            "local_outside_in": local_scan,
            "local_project_validator_exact": bool(is_palindrome(CANDIDATE)),
            "exact_center_span": local_scan["exact"] and bool(is_palindrome(CANDIDATE)),
            "shortcut_debt": {
                "whole_token_reversal_pairs": [list(pair) for pair in token_reversals],
                "self_palindromic_tokens": self_pal,
                "singleton_tokens": singleton,
                "repeated_tokens": repeated,
            },
        },
        "attempted_full_rendering": attempted,
        "full_candidate_audit": {
            "letters": len(full_tape),
            "growth": len(full_tape) - len(parent_tape),
            "outside_in": full_scan,
            "project_validator_exact": project_exact,
            "forward_sha256": fwd,
            "reverse_sha256": rev,
            "hashes_equal": fwd == rev,
            "exact": full_scan["exact"] and project_exact and fwd == rev,
            "admitted": False,
            "reason": "Local span first mismatches at offset 5: h/e; full tape first mismatches at normalized offsets 289/304. Center insertion therefore does not preserve exactness.",
        },
        "reader_status": "Not presented to readers; no readability claim.",
        "next_operator": "Do not repair this line by phrase substitution: its proposed two event clauses do not satisfy the character equation. Next attempt must derive a fresh clause from an explicit letter-residual constraint, then verify local closure before touching the 568 parent.",
    }


if __name__ == "__main__":
    result = main()
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "local_letters": result["construction"]["local_outside_in"]["letters"],
        "local_exact": result["construction"]["exact_center_span"],
        "local_mismatch": result["construction"]["local_outside_in"]["first_mismatch"],
        "full_letters": result["full_candidate_audit"]["letters"],
        "full_exact": result["full_candidate_audit"]["exact"],
        "full_mismatch": result["full_candidate_audit"]["outside_in"]["first_mismatch"],
    }, sort_keys=True))
