"""Record one bounded promise/fulfillment center-span preflight."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/luna6-promise-fulfillment-preflight-20260923.json"
PARENT = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
PREFLIGHT_REVISION = "67b671f9"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
CANDIDATE = "I vowed to help her today; I kept my word."


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def main() -> dict[str, object]:
    sys.path.insert(0, str(ROOT))
    from llm_palindrome.validator import is_palindrome

    parent_json = json.loads(PARENT.read_text())
    parent = next(row["rendered"] for row in parent_json["rows"]
                  if row["id"] == "outer-causal-scene-568-working-incumbent")
    parent_tape = letters(parent)
    if len(parent_tape) != 568 or hashlib.sha256(parent_tape.encode()).hexdigest() != PARENT_SHA256:
        raise AssertionError("pinned 568 parent identity changed")
    if not is_palindrome(parent):
        raise AssertionError("pinned 568 parent is no longer exact")

    grep = subprocess.run(
        ["git", "grep", "-F", CANDIDATE, PREFLIGHT_REVISION,
         "--", "experiments", "runs", "docs", "data"],
        cwd=ROOT, capture_output=True, text=True)
    if grep.returncode not in (0, 1):
        raise RuntimeError(grep.stderr)
    tape = letters(CANDIDATE)
    reverse = tape[::-1]
    mismatch = next((i for i, (left, right) in enumerate(zip(tape, reverse))
                     if left != right), None)
    tokens = [word.strip(".,;:?!\"'").casefold() for word in CANDIDATE.split()]
    return {
        "experiment_id": "luna6-promise-fulfillment-preflight-20260923",
        "status": "rejected_exactness_at_first_character",
        "method_signature": "one authored promise-then-fulfillment scene at the 568 midpoint",
        "novelty_preflight": {
            "candidate_absent_from_tracked_preflight_revision": grep.returncode == 1,
            "revision": PREFLIGHT_REVISION,
            "literal": CANDIDATE,
            "scope": "literal-level preflight only; no claim that promise/fulfillment is a novel grammar family",
        },
        "parent": {"artifact": str(PARENT.relative_to(ROOT)), "letters": 568,
                   "sha256": PARENT_SHA256, "exact": True},
        "construction": {
            "rendered": CANDIDATE,
            "normalized": tape,
            "reverse": reverse,
            "letters": len(tape),
            "left_clause_parse": "I (subject) vowed (past verb) to help her today (infinitival complement).",
            "right_clause_parse": "I (subject) kept (past verb) my word (object).",
            "semantic_relation": "the speaker promises assistance, then asserts fulfillment",
            "outside_in_exact": tape == reverse,
            "project_validator_exact": bool(is_palindrome(CANDIDATE)),
            "first_mismatch_cursor": mismatch,
            "first_mismatch": {"left": tape[mismatch], "right": reverse[mismatch]}
            if mismatch is not None else None,
            "wrapped_parent": False,
            "whole_token_reversal_pairs": sorted({(a, b) for a in tokens for b in tokens
                                                   if a != b and a[::-1] == b}),
            "repeated_tokens": sorted({word for word in set(tokens) if tokens.count(word) > 1}),
            "self_palindromic_tokens": sorted({word for word in tokens if word == word[::-1]}),
        },
        "next_operator": (
            "Leave free-form midpoint generation. Choose one untried partial-word seam in the pinned 568 tape, "
            "record its owner and required residual first, then lexicalize one complete event from that exact "
            "character obligation; if its first character conflicts, persist the cursor and change seams."
        ),
    }


if __name__ == "__main__":
    result = main()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "letters": result["construction"]["letters"],
        "exact": result["construction"]["outside_in_exact"],
        "first_mismatch_cursor": result["construction"]["first_mismatch_cursor"],
        "novel_literal": result["novelty_preflight"]["candidate_absent_from_tracked_preflight_revision"],
    }, sort_keys=True))
