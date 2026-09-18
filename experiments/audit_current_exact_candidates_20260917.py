"""Repository-wide audit of current exact candidate renderings.

This is an evidence collector, not a readability classifier.  It independently
normalizes every compact string in ``runs/**/*.json``, checks exact reversal,
then records the central construction gate and lightweight lexical diagnostics.
Only blinded readers can certify that any row is readable.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "current-exact-candidate-audit-20260917.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize


def walk(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, f"{path}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, f"{path}/{index}")


def independent_exact(tape: str) -> bool:
    """Two-pointer check kept separate from the admission implementation."""
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            return False
        left += 1
        right -= 1
    return True


def lexical_diagnostics(words: tuple[str, ...]) -> dict[str, float]:
    frequencies = [zipf_frequency(word, "en") for word in words]
    return {
        "known_word_rate": round(sum(freq >= 2.5 for freq in frequencies) / max(1, len(words)), 4),
        "common_word_rate": round(sum(freq >= 3.5 for freq in frequencies) / max(1, len(words)), 4),
        "mean_zipf": round(sum(frequencies) / max(1, len(frequencies)), 4),
        "short_word_rate": round(sum(len(word) <= 2 for word in words) / max(1, len(words)), 4),
        "content_repeat_rate": round((len(words) - len(set(words))) / max(1, len(words)), 4),
    }


def collect() -> list[dict[str, Any]]:
    by_tape: dict[str, dict[str, Any]] = {}
    for path in sorted(RUNS.rglob("*.json")):
        if path == OUT:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for json_path, text in walk(payload):
            words = tokenize(text)
            if len(words) < 5 or len(text) > 2_000:
                continue
            try:
                tape = normalize_letters(text)
            except ValueError:
                continue
            if not 40 <= len(tape) <= 300 or not independent_exact(tape):
                continue
            diagnostics = lexical_diagnostics(words)
            checks = mechanical_admission_checks(text, min_letters=40, max_letters=300)
            row = {
                "rendered": text,
                "letters": len(tape),
                "normalized_letters": tape,
                "sha256": hashlib.sha256(tape.encode()).hexdigest(),
                "provenance": {"file": str(path.relative_to(ROOT)), "json_path": json_path},
                "words": list(words),
                "independent_exact": True,
                "mechanical_checks": checks,
                "central_mechanical_pass": all(checks.values()),
                "lexical_diagnostics": diagnostics,
                "reader_status": "not_run",
                "reader_worthy": False,
            }
            prior = by_tape.get(tape)
            if prior is None:
                by_tape[tape] = row
            else:
                prior.setdefault("additional_provenance", []).append(row["provenance"])
    rows = list(by_tape.values())
    rows.sort(key=lambda row: (
        row["central_mechanical_pass"],
        row["lexical_diagnostics"]["mean_zipf"],
        row["lexical_diagnostics"]["known_word_rate"],
        row["letters"],
    ), reverse=True)
    return rows


def main() -> None:
    rows = collect()
    admitted = [row for row in rows if row["central_mechanical_pass"]]
    report = {
        "status": "audit_complete_no_reader_promotion",
        "method": {
            "source": "current runs/**/*.json strings",
            "threshold": "40-300 ASCII letters, at least five tokens",
            "exactness": "independent two-pointer comparison after ASCII-letter normalization",
            "anti_shortcut": "central mechanical_admission_checks; programmatic checks do not certify readability",
            "reader_protocol": "not run; candidates require randomized blinded intact-prose review",
        },
        "counts": {"unique_exact_candidates": len(rows), "central_mechanical_pass": len(admitted)},
        "candidates": rows[:200],
        "conclusion": (
            "No row is reader-worthy by this audit. Rendered text, exact tape, and provenance are preserved; "
            "the next step for any mechanically passing row is blinded human review, not promotion by proxy score."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"], indent=2))
    for row in rows[:20]:
        print(f"{row['letters']:3d} mech={row['central_mechanical_pass']} mean_zipf={row['lexical_diagnostics']['mean_zipf']:.2f} :: {row['rendered']}")


if __name__ == "__main__":
    main()
