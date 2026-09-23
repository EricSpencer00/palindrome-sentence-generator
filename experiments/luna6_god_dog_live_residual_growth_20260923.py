"""Verify and record a distinct live-residual repair of the pinned 568 tape."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUTPUT_PATH = ROOT / "runs/luna6-god-dog-live-residual-growth-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
PREFLIGHT_REVISION = "79274ec9"
LEFT_CUT = 16
RIGHT_CUT = 548
LEFT_INSERT = " Nora. Nora saw God. Nadia saw God. Dog was"
RIGHT_INSERT = ". Aron saw God. Dog was Aidan. Dog was Aron."
NOVELTY_PHRASES = (
    "Nora saw God",
    "Nadia saw God",
    "Dog was Nora",
    "Aron saw God",
    "Dog was Aidan",
    "Dog was Aron",
)


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


def outside_in(tape: str) -> dict[str, object]:
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


def sha(tape: str) -> str:
    return hashlib.sha256(tape.encode("ascii")).hexdigest()


def git_literal_absent(phrase: str) -> bool:
    result = subprocess.run(
        ["git", "grep", "-F", phrase, PREFLIGHT_REVISION, "--",
         "experiments", "runs", "docs", "data"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return result.returncode == 1


def main() -> dict[str, object]:
    sys.path.insert(0, str(ROOT))
    from llm_palindrome.validator import is_palindrome

    payload = json.loads(PARENT_PATH.read_text())
    parent = next(
        row["rendered"] for row in payload["rows"]
        if row["id"] == "outer-causal-scene-568-working-incumbent"
    )
    parent_tape = letters(parent)
    if len(parent_tape) != 568 or sha(parent_tape) != PARENT_SHA256:
        raise AssertionError("pinned 568 parent identity changed")
    if not outside_in(parent_tape)["exact"] or not is_palindrome(parent):
        raise AssertionError("pinned 568 parent failed exact validation")

    left_raw = raw_after_letters(parent, LEFT_CUT)
    right_raw = raw_after_letters(parent, RIGHT_CUT)
    if (left_raw, right_raw) != (20, 758) or parent[right_raw] != ".":
        raise AssertionError("pinned seam punctuation/cursors changed")

    # The insertion's terminal period replaces the incumbent period at the
    # right cursor; the retained suffix begins with the existing space + Aron.
    rendered = (
        parent[:left_raw] + LEFT_INSERT + parent[left_raw:right_raw]
        + RIGHT_INSERT + parent[right_raw + 1:]
    )
    tape = letters(rendered)

    prefix = parent_tape[:LEFT_CUT]
    left_residual = parent_tape[LEFT_CUT:LEFT_CUT + 4]
    retained_middle = parent_tape[LEFT_CUT + 4:RIGHT_CUT]
    right_residual = parent_tape[RIGHT_CUT:RIGHT_CUT + 4]
    suffix = parent_tape[RIGHT_CUT + 4:]
    left_emission, right_emission = letters(LEFT_INSERT), letters(RIGHT_INSERT)
    equation = left_emission + left_residual == left_residual + right_emission[::-1]
    closure = (
        prefix == suffix[::-1]
        and left_residual == right_residual[::-1]
        and retained_middle == retained_middle[::-1]
        and equation
    )
    novelty = {phrase: git_literal_absent(phrase) for phrase in NOVELTY_PHRASES}
    scan = outside_in(tape)
    project_exact = bool(is_palindrome(rendered))
    forward_sha, reverse_sha = sha(tape), sha(tape[::-1])
    exact = scan["exact"] and project_exact and forward_sha == reverse_sha

    if not all(novelty.values()):
        raise AssertionError("an authored clause literal already exists in preflight history")
    if not closure or not exact or len(tape) != 630:
        raise AssertionError("residual closure or independent full-tape validation failed")

    return {
        "experiment_id": "luna6-god-dog-live-residual-growth-20260923",
        "status": "exact_630_letter_working_child_with_explicit_repair_debt",
        "provenance": {
            "parent_artifact": str(PARENT_PATH.relative_to(ROOT)),
            "parent_id": "outer-causal-scene-568-working-incumbent",
            "parent_letters": len(parent_tape),
            "parent_sha256": PARENT_SHA256,
            "construction": "one authored insertion at the pinned nora/aron partial-word seam",
            "borrowed_catalogue_text": False,
            "preflight_revision": PREFLIGHT_REVISION,
            "exact_clause_literals_absent": novelty,
        },
        "live_seam": {
            "normalized_cursors": [LEFT_CUT, RIGHT_CUT],
            "raw_cursors": [left_raw, right_raw],
            "left_residual": left_residual,
            "right_residual": right_residual,
            "retained_middle_letters": len(retained_middle),
            "left_emission": left_emission,
            "right_emission": right_emission,
            "equation": "left_emission + nora = nora + reverse(right_emission)",
            "equation_exact": equation,
            "residual_closed": closure,
            "character_contradictions": 0,
        },
        "inserted_surface": {
            "left": LEFT_INSERT.strip(),
            "right": RIGHT_INSERT.strip(" ."),
            "new_clause_sequence": [
                "Nora saw God.",
                "Nadia saw God.",
                "Dog was Nora.",
                "Aron saw God.",
                "Dog was Aidan.",
                "Dog was Aron.",
            ],
            "semantic_note": "distinct observation/identity clauses; their discourse linkage remains rough",
        },
        "rendered_full_text": rendered,
        "repair_debt": {
            "whole_token_reversal_pairs": [["dog", "god"]],
            "repeated_tokens": ["dog", "was", "saw", "god"],
            "repeated_clause_frames": ["Nora/Nadia/Aron saw God", "Dog was Nora/Aidan/Aron"],
            "syntactic_flags": [
                "the identity assertions Dog was X need narrative grounding",
                "the inserted scene remains conspicuously symmetry-driven",
                "the parent retains substantial rough syntax and recycled motifs",
            ],
            "human_readability_validation": "not performed; no readability claim",
        },
        "validation": {
            "letters": len(tape),
            "growth_over_568": len(tape) - len(parent_tape),
            "outside_in": scan,
            "project_validator_exact": project_exact,
            "forward_sha256": forward_sha,
            "reverse_sha256": reverse_sha,
            "hashes_equal": forward_sha == reverse_sha,
            "exact": exact,
        },
        "next_repair": "Do not add another God/dog or Dog-was identity frame. Preserve this exact child while searching a different seam/event frame for coherent new content, then use the actual candidate in blinded intact-versus-shuffled human ratings.",
    }


if __name__ == "__main__":
    result = main()
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "letters": result["validation"]["letters"],
        "growth": result["validation"]["growth_over_568"],
        "exact": result["validation"]["exact"],
        "sha256": result["validation"]["forward_sha256"],
        "residual_closed": result["live_seam"]["residual_closed"],
        "rendered_full_text": result["rendered_full_text"],
    }, sort_keys=True))
