"""Bounded character-ledger authoring probe (no catalogue/Brown material)."""
import glob
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ROOT = Path(__file__).parents[1]
LEFTS = [
    "Careful makers restore old radios.",
    "Bright students solve hard puzzles.",
    "Patient nurses record each dosage.",
    "Quiet artists frame winter scenes.",
]
RIGHT_GUESSES = ["So does the team.", "Then the work ends.", "The notes remain.", "Before dawn."]


def fingerprint(output=None):
    """Return normalized strings from the repository before this run writes output."""
    seen = set()
    output = Path(output).resolve() if output else None
    paths = glob.glob(str(ROOT / "data" / "*.json"))
    paths += glob.glob(str(ROOT / "runs" / "**" / "*.json*"), recursive=True)
    for raw_path in paths:
        path = Path(raw_path)
        if output and path.resolve() == output:
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue

        def walk(value):
            if isinstance(value, str):
                try:
                    seen.add(normalize_letters(value))
                except ValueError:
                    pass
            elif isinstance(value, dict):
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(payload)
    return seen


def main():
    output = ROOT / "runs" / "luna_character_ledger_promptbank_20260915.json"
    known = fingerprint(output)
    rows = []
    for left in LEFTS:
        tape = normalize_letters(left)
        target = tape[::-1]
        for right in RIGHT_GUESSES:
            text = left.rstrip(".") + " " + right
            candidate_tape = normalize_letters(text)
            checks = mechanical_admission_checks(text, min_letters=39, max_letters=180)
            rows.append(
                {
                    "left_clause": left,
                    "reversed_tape_constraint": target,
                    "right_clause": right,
                    "rendered": text,
                    "tape": candidate_tape,
                    "exact": candidate_tape == candidate_tape[::-1],
                    "known_tape": candidate_tape in known,
                    "checks": checks,
                    "admitted": all(checks.values()),
                }
            )
    result = {
        "status": "complete_character_ledger_promptbank",
        "state_space_signature": hashlib.sha256(
            b"char-ledger-v2|4 natural left clauses|16 independently authored right guesses|reverse-tape constraint"
        ).hexdigest(),
        "repository_tapes": len(known),
        "proposals": rows,
        "next_operator": "Use local gpt-oss constrained decoding to author right clauses while exposing only reverse-tape-compatible prefixes.",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print({"proposals": len(rows), "exact": sum(x["exact"] for x in rows), "admitted": sum(x["admitted"] for x in rows)})


if __name__ == "__main__":
    main()
