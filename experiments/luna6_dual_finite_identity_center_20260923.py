"""One incumbent-specific, residual-closed finite-clause identity span."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUTPUT_PATH = ROOT / "runs/luna6-dual-finite-identity-center-20260923.json"
PARENT_SHA = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
CANDIDATE = "Aidan saw Nadia; Aidan was Nadia."
EXPECTED_LOCAL = "aidansawnadiaaidanwasnadia"


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def raw_offset(text: str, target: int) -> int:
    count = 0
    for i, char in enumerate(text):
        if char.isascii() and char.isalpha():
            if count == target:
                return i
            count += 1
    if count == target:
        return len(text)
    raise ValueError(target)


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


def main() -> dict:
    sys.path.insert(0, str(ROOT))
    from llm_palindrome.validator import is_palindrome

    parent_json = json.loads(PARENT_PATH.read_text())
    parent = next(row["rendered"] for row in parent_json["rows"]
                  if row["id"] == "outer-causal-scene-568-working-incumbent")
    parent_tape = letters(parent)
    if len(parent_tape) != 568 or sha(parent_tape) != PARENT_SHA:
        raise AssertionError("pinned 568 parent identity changed")
    if not scan(parent_tape)["exact"] or not is_palindrome(parent):
        raise AssertionError("pinned 568 parent failed exact validation")

    grep = subprocess.run(
        ["git", "grep", "-F", CANDIDATE, "HEAD", "--", "experiments", "runs", "docs", "data"],
        cwd=ROOT, capture_output=True, text=True)
    if grep.returncode not in (0, 1):
        raise RuntimeError(grep.stderr)
    novel_literal = grep.returncode == 1

    local = letters(CANDIDATE)
    if local != EXPECTED_LOCAL:
        raise AssertionError("candidate tape differed from the recorded live equation")
    local_scan = scan(local)
    local_validator = bool(is_palindrome(CANDIDATE))
    if not (novel_literal and local_scan["exact"] and local_validator):
        raise AssertionError("novelty or local exact gate failed; do not wrap")

    center = 284
    raw = raw_offset(parent, center)
    full = parent[:raw] + CANDIDATE + " " + parent[raw:]
    full_tape = letters(full)
    full_scan = scan(full_tape)
    project_exact = bool(is_palindrome(full))
    forward, reverse = sha(full_tape), sha(full_tape[::-1])
    tokens = [part.lower() for part in CANDIDATE.replace(";", "").replace(".", "").split()]
    token_reversals = sorted({(a, b) for a in tokens for b in tokens
                              if a != b and a[::-1] == b})
    repeated = sorted({word for word in set(tokens) if tokens.count(word) > 1})
    self_pal = sorted({word for word in tokens if word == word[::-1]})
    singleton = sorted({word for word in tokens if len(word) == 1})
    exact = full_scan["exact"] and project_exact and forward == reverse

    return {
        "experiment_id": "luna6-dual-finite-identity-center-20260923",
        "status": "exact_594_letter_working_child_with_explicit_shortcut_debt" if exact
                  else "full_render_failed_exact_audit",
        "novelty_preflight": {
            "literal_absent_from_tracked_HEAD": novel_literal,
            "literal": CANDIDATE,
            "scope": "one specific event/identity instantiation at the pinned parent midpoint; no claim that the general topology is novel",
            "parent": {"letters": 568, "sha256": PARENT_SHA, "exact": True},
            "center_offset": center,
        },
        "local_construction": {
            "rendered": CANDIDATE,
            "normalized": local,
            "reverse": local[::-1],
            "letters": len(local),
            "live_equation": {
                "left_clause_tape": "aidansaw",
                "center_boundary_residual": "nadia",
                "right_clause_tape": "aidanwasnadia",
                "complete_local_tape": "aidansawnadiaaidanwasnadia",
            },
            "parse": [
                {"text": "Aidan saw Nadia", "roles": {"agent": "Aidan", "action": "saw", "theme": "Nadia"}},
                {"text": "Aidan was Nadia", "roles": {"subject": "Aidan", "copula": "was", "identity": "Nadia"}},
            ],
            "semantic_relation": "identity/disguise reading: the perceiver in the first event is identified with the person named in the second clause; unusual and not reader-validated",
            "outside_in": local_scan,
            "project_validator_exact": local_validator,
            "forward_sha256": sha(local),
            "reverse_sha256": sha(local[::-1]),
            "exact": local_scan["exact"] and local_validator and local == local[::-1],
        },
        "rendered_full_tape": full,
        "full_audit": {
            "letters": len(full_tape), "growth": len(full_tape) - len(parent_tape),
            "outside_in": full_scan,
            "project_validator_exact": project_exact,
            "forward_sha256": forward, "reverse_sha256": reverse,
            "hashes_equal": forward == reverse,
            "exact": exact,
            "admitted_working_child": exact,
        },
        "shortcut_and_readability_debt": {
            "whole_token_reversal_pairs": [list(pair) for pair in token_reversals],
            "repeated_tokens": repeated,
            "self_palindromic_tokens": self_pal,
            "singleton_tokens": singleton,
            "reader_status": "No blinded ratings; the exact extension is not a readability claim. The identity/disguise relation and duplicated frame need human review.",
        },
        "frontier_note": "Keep the exact 568 parent and 589 child; this 594-letter child is longer than both but has visible lexical/semantic debt and is not reader-worthy proof.",
        "next_repair": "Replace the aligned name/verb reverse pairs with an offset lexical boundary while preserving a complete identity or perception relation; carry the residual before selecting surface words.",
    }


if __name__ == "__main__":
    result = main()
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "local_letters": result["local_construction"]["letters"],
        "local_exact": result["local_construction"]["exact"],
        "full_letters": result["full_audit"]["letters"],
        "growth": result["full_audit"]["growth"],
        "full_exact": result["full_audit"]["exact"],
        "token_reversal_debt": result["shortcut_and_readability_debt"]["whole_token_reversal_pairs"],
        "repeated_tokens": result["shortcut_and_readability_debt"]["repeated_tokens"],
    }, sort_keys=True))
