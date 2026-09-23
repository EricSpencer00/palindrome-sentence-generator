"""One offset-boundary finite-complement realization and its first cursor failure."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/luna6-offset-boundary-complement-obstruction-20260923.json"
LEFT = "An aide made a portrait"
RIGHT = "Nora heard that an aide became Diana"
RENDER = LEFT + "; " + RIGHT + "."


def letters(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def scan(value: str) -> dict:
    i, j = 0, len(value) - 1
    while i < j and value[i] == value[j]:
        if value[i] != value[j]:
            break
        i += 1
        j -= 1
    exact = i >= j
    return {
        "exact": exact,
        "letters": len(value),
        "matched_outer_pairs": i,
        "first_mismatch": None if exact else {
            "offset_left": i, "offset_right": j,
            "left": value[i], "right": value[j],
        },
        "left_residual": value[i:i + 36],
        "right_reverse_residual": value[max(0, j - 35):j + 1][::-1],
    }


def main() -> dict:
    left, right = letters(LEFT), letters(RIGHT)
    tape = left + right
    audit = scan(tape)
    grep = subprocess.run(
        ["git", "grep", "-F", RENDER, "HEAD", "--", "experiments", "runs", "docs", "data"],
        cwd=ROOT, capture_output=True, text=True)
    if grep.returncode not in (0, 1):
        raise RuntimeError(grep.stderr)
    if audit["first_mismatch"] != {"offset_left": 8, "offset_right": 40,
                                   "left": "d", "right": "c"}:
        raise AssertionError("recorded residual changed")
    return {
        "experiment_id": "luna6-offset-boundary-complement-obstruction-20260923",
        "status": "complete_clause_pair_rejected_at_live_cursor",
        "novelty_preflight": {
            "exact_literal_absent_from_tracked_HEAD": grep.returncode == 1,
            "literal": RENDER,
            "claim_scope": "one specific Aide/Diana offset-boundary realization only; generic finite-complement search is already represented and not claimed novel",
        },
        "construction": {
            "topology": "left subject `An aide` crosses the right suffix boundary `e Diana`; right clause is a finite hearing complement",
            "left_clause": {"surface": LEFT,
                            "parse": "An aide (agent) made (past transitive verb) a portrait (theme)."},
            "right_clause": {"surface": RIGHT,
                             "parse": "Nora (matrix subject) heard (matrix verb) that [an aide (embedded subject) became (past copula) Diana (identity complement)]."},
            "semantic_relation": "Nora hears an identity claim about the aide; the same aide is the agent of the portrait-making event.",
            "outer_alignment": {"left_opening": "anaide", "right_suffix": "eDiana",
                                "reverse_suffix": "anaide", "matched_initial_letters": 6},
            "left_tape": left,
            "right_tape": right,
            "reverse_right_tape": right[::-1],
            "full_local_tape": tape,
            "letters": len(tape),
            "outside_in": audit,
            "forward_sha256": hashlib.sha256(tape.encode("ascii")).hexdigest(),
            "reverse_sha256": hashlib.sha256(tape[::-1].encode("ascii")).hexdigest(),
            "exact": audit["exact"],
            "first_residual": {
                "cursor": 8,
                "left_next": "d" ,
                "right_reverse_next": "c",
                "left_residual": audit["left_residual"],
                "right_reverse_residual": audit["right_reverse_residual"],
            },
            "wrapped_parent": False,
            "reason": "After the useful `anaide` offset match and two more letters, `made` begins `da…` while the reversed remainder of `became` begins `ca…`; grammar cannot alter the live d/c obligation without changing this realization.",
        },
        "next_axis": {
            "name": "compound-boundary resegmentation",
            "operator": "At one actual 568 seam, assign the residual across a freshly authored compound whose internal morpheme boundary is parsed as a word boundary on the reflected side; use complete event roles but no finite complement, insertion wrapper, or aligned reversed tokens.",
            "preflight_required": "Check the exact compound signature against tracked history before choosing its one lexicalization.",
        },
    }


if __name__ == "__main__":
    result = main()
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "letters": result["construction"]["letters"],
        "exact": result["construction"]["exact"],
        "cursor": result["construction"]["first_residual"]["cursor"],
        "mismatch": result["construction"]["outside_in"]["first_mismatch"],
    }, sort_keys=True))
