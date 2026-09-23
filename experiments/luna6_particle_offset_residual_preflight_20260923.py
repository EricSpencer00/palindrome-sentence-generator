"""Record one boundary-shifted particle residual that closes only as a fragment."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/luna6-particle-offset-residual-preflight-20260923.json"
LEFT = "No, put up"
RIGHT = "put upon"
RENDER = "No, put up; put upon."


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def scan(value: str) -> dict:
    i, j = 0, len(value) - 1
    while i < j and value[i] == value[j]:
        i += 1
        j -= 1
    return {
        "exact": i >= j,
        "letters": len(value),
        "matched_outer_pairs": i,
        "first_mismatch": None if i >= j else {
            "offset_left": i, "offset_right": j,
            "left": value[i], "right": value[j],
        },
        "left_residual": value[i:i + 30],
        "right_reverse_residual": value[max(0, j - 29):j + 1][::-1],
    }


def main() -> dict:
    left, right, full = letters(LEFT), letters(RIGHT), letters(RENDER)
    reverse_residual = left[::-1]
    grep = subprocess.run(
        ["git", "grep", "-F", RENDER, "HEAD", "--", "experiments", "runs", "docs", "data"],
        cwd=ROOT, capture_output=True, text=True)
    if grep.returncode not in (0, 1):
        raise RuntimeError(grep.stderr)
    return {
        "experiment_id": "luna6-particle-offset-residual-preflight-20260923",
        "status": "exact_letter_closure_rejected_at_local_grammar_gate",
        "novelty_preflight": {
            "literal_absent_from_tracked_HEAD": grep.returncode == 1,
            "rendered_literal": RENDER,
            "claim_scope": "one incumbent-independent grammar preflight; not a generation-method claim",
        },
        "attempt": {
            "representation": "boundary-shifted phrasal-particle residual",
            "left_surface": LEFT,
            "left_tape": left,
            "reverse_character_residual": reverse_residual,
            "right_surface": RIGHT,
            "right_tape": right,
            "residual_closes": left == right[::-1],
            "full_local_render": RENDER,
            "local_tape": full,
            "outside_in": scan(full),
            "sha_forward": hashlib.sha256(full.encode("ascii")).hexdigest(),
            "sha_reverse": hashlib.sha256(full[::-1].encode("ascii")).hexdigest(),
            "exact": full == full[::-1],
            "parse_obstruction": "`put upon` is incomplete here: `upon` has no object or licensed complement, and `put up` has no object. The line is two fragments/commands, not a complete event sentence with a semantic theme.",
            "shortcut_debt": {"repeated_tokens": ["put"], "whole_token_reversal_pairs": [],
                              "self_palindromic_tokens": [], "singleton_tokens": []},
            "parent_rendered": False,
            "admitted": False,
        },
        "next_axis": "Finite-complement boundary resegmentation: choose one transitive event frame, then require the reversed suffix to parse as a distinct complete finite clause across a shifted word boundary; do not reuse particle attachment or the No/put frame.",
    }


if __name__ == "__main__":
    result = main()
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "local_letters": result["attempt"]["outside_in"]["letters"],
        "local_exact": result["attempt"]["exact"],
        "first_mismatch": result["attempt"]["outside_in"]["first_mismatch"],
        "admitted": result["attempt"]["admitted"],
    }, sort_keys=True))
