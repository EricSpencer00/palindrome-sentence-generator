"""One residual-owned, partial-word seam extension on the pinned 568 tape."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUTPUT_PATH = ROOT / "runs/luna6-nora-aron-live-residual-growth-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
PREFLIGHT_REVISION = "393b6c5b"
LEFT_CUT = 16
RIGHT_CUT = 548
LEFT_INSERT = " Nora. Nora sees Aron. Aron sees"
RIGHT_INSERT = ". Aron sees Nora. Nora sees Aron."


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def raw_after_letters(text: str, count: int) -> int:
    seen = 0
    for index, char in enumerate(text):
        if char.isascii() and char.isalpha():
            seen += 1
            if seen == count:
                return index + 1
    if count == 0:
        return 0
    raise ValueError(f"surface contains fewer than {count} letters")


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
            "offset_left": left,
            "offset_right": right,
            "left": tape[left],
            "right": tape[right],
        },
        "left_residual": tape[left:left + 40],
        "right_reverse_residual": tape[max(0, right - 39):right + 1][::-1],
    }


def sha(tape: str) -> str:
    return hashlib.sha256(tape.encode("ascii")).hexdigest()


def main() -> dict[str, object]:
    sys.path.insert(0, str(ROOT))
    from llm_palindrome.validator import is_palindrome

    payload = json.loads(PARENT_PATH.read_text())
    parent = next(row["rendered"] for row in payload["rows"]
                  if row["id"] == "outer-causal-scene-568-working-incumbent")
    parent_tape = letters(parent)
    if len(parent_tape) != 568 or sha(parent_tape) != PARENT_SHA256:
        raise AssertionError("pinned 568 parent identity changed")
    if not scan(parent_tape)["exact"] or not is_palindrome(parent):
        raise AssertionError("pinned 568 parent failed exact validation")

    # Exact literal preflight against the committed tree before this experiment.
    novelty_phrases = ["Nora sees Aron", "Aron sees Nora"]
    novelty = {}
    for phrase in novelty_phrases:
        result = subprocess.run(
            ["git", "grep", "-F", phrase, PREFLIGHT_REVISION,
             "--", "experiments", "runs", "docs", "data"],
            cwd=ROOT, capture_output=True, text=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        novelty[phrase] = result.returncode == 1

    left_raw = raw_after_letters(parent, LEFT_CUT)
    right_raw = raw_after_letters(parent, RIGHT_CUT)
    if (left_raw, right_raw) != (20, 758):
        raise AssertionError("pinned raw seam offsets changed")
    # The right cursor is immediately before the incumbent's period after Aidan;
    # replace that punctuation with the sentence boundary for the new clauses.
    rendered = (parent[:left_raw] + LEFT_INSERT + parent[left_raw:right_raw]
                + RIGHT_INSERT + parent[right_raw + 1:])
    tape = letters(rendered)
    project_exact = bool(is_palindrome(rendered))
    full_scan = scan(tape)
    forward, reverse = sha(tape), sha(tape[::-1])

    prefix = parent_tape[:LEFT_CUT]
    left_residual = parent_tape[LEFT_CUT:LEFT_CUT + 4]
    middle = parent_tape[LEFT_CUT + 4:RIGHT_CUT]
    right_residual = parent_tape[RIGHT_CUT:RIGHT_CUT + 4]
    suffix = parent_tape[RIGHT_CUT + 4:]
    left_emission, right_emission = letters(LEFT_INSERT), letters(RIGHT_INSERT)
    residual_equation = (
        left_emission + left_residual
        == left_residual + right_emission[::-1]
    )
    decomposition_exact = (
        prefix == suffix[::-1]
        and left_residual == right_residual[::-1]
        and middle == middle[::-1]
        and residual_equation
    )

    added_surface = LEFT_INSERT + RIGHT_INSERT
    added_tokens = [word.strip(".,;:?!\"'").casefold()
                    for word in added_surface.split()]
    reversed_token_pairs = sorted({(a, b) for a in added_tokens for b in added_tokens
                                   if a != b and a[::-1] == b})
    repeated_tokens = sorted({word for word in set(added_tokens)
                              if added_tokens.count(word) > 1})
    self_palindromic_tokens = sorted({word for word in added_tokens if word == word[::-1]})

    exact = full_scan["exact"] and project_exact and forward == reverse
    if not all(novelty.values()):
        raise AssertionError("one or more candidate clauses already appear in preflight history")
    if not decomposition_exact or not exact or len(tape) != 616:
        raise AssertionError("residual closure or independent full-tape validation failed")

    return {
        "experiment_id": "luna6-nora-aron-live-residual-growth-20260923",
        "status": "exact_616_letter_working_child_with_explicit_repair_debt",
        "novelty_preflight": {
            "revision": PREFLIGHT_REVISION,
            "exact_clause_literals_absent": novelty,
            "scope": "one authored seam-specific realization; no broad operator-family novelty claim",
        },
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "id": "outer-causal-scene-568-working-incumbent",
            "letters": len(parent_tape),
            "sha256": PARENT_SHA256,
            "exact": True,
        },
        "live_seam": {
            "normalized_cursors": [LEFT_CUT, RIGHT_CUT],
            "raw_cursors": [left_raw, right_raw],
            "prefix_tape": prefix,
            "left_residual": left_residual,
            "retained_middle_letters": len(middle),
            "retained_middle_exact": middle == middle[::-1],
            "right_residual": right_residual,
            "suffix_tape": suffix,
            "left_emission": left_emission,
            "right_emission": right_emission,
            "equation": "left_emission + nora = nora + reverse(right_emission)",
            "equation_exact": residual_equation,
            "residual_closed": decomposition_exact,
            "character_contradictions": 0,
        },
        "authored_clause_sequences": {
            "left": LEFT_INSERT.strip(),
            "left_parse": [
                "Nora (subject) sees (finite verb) Aron (object).",
                "Aron (subject) sees (finite verb; object supplied by the following Nora).",
            ],
            "right": RIGHT_INSERT.strip(" ."),
            "right_parse": [
                "Aron (subject) sees (finite verb) Nora (object).",
                "Nora (subject) sees (finite verb) Aron (object).",
            ],
            "semantic_reading": "reciprocal seeing events between Nora and Aron; repetition and weak discourse integration remain repair debt",
        },
        "rendered_full_tape": rendered,
        "shortcut_and_readability_debt": {
            "whole_token_reversal_pairs_in_added_surface": [list(pair) for pair in reversed_token_pairs],
            "repeated_tokens_in_added_surface": repeated_tokens,
            "self_palindromic_tokens_in_added_surface": self_palindromic_tokens,
            "repeated_reciprocal_clause_shell": True,
            "reader_status": "no human ratings; exactness is not a readability claim",
        },
        "full_audit": {
            "letters": len(tape),
            "growth_over_568": len(tape) - len(parent_tape),
            "outside_in": full_scan,
            "project_validator_exact": project_exact,
            "forward_sha256": forward,
            "reverse_sha256": reverse,
            "hashes_equal": forward == reverse,
            "exact": exact,
            "admitted_working_child": exact,
        },
        "next_repair": "Preserve this exact residual closure, then replace the repeated reciprocal sees shell with a distinct event frame at the same seam; do not grow by repeating Nora/Aron clauses again.",
    }


if __name__ == "__main__":
    result = main()
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "letters": result["full_audit"]["letters"],
        "growth": result["full_audit"]["growth_over_568"],
        "exact": result["full_audit"]["exact"],
        "sha256": result["full_audit"]["forward_sha256"],
        "residual_equation": result["live_seam"]["equation_exact"],
    }, sort_keys=True))
