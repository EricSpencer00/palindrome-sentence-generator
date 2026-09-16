"""Record and independently audit one bounded RhythmAI authoring probe.

The text is the exact ANSI-stripped first line captured from the local
``ollama run imetaexabeam/RhythmAI:27b`` command.  This file deliberately does
not claim that a model response is reproducible or readable; it makes the
negative repair evidence explicit and mechanically checkable.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "rhythmai-authoring-probe-20260916.json"
MODEL = "imetaexabeam/RhythmAI:27b"
PROMPT = (
    "Write one original, ordinary-English sentence of at least 80 letters. "
    "After lowercasing and removing spaces and punctuation, the letters must "
    "be an exact palindrome. Use distinct content words, one coherent scene, "
    "normal grammar, no repeated clauses, no famous or catalogue palindrome, "
    "and output only the sentence."
)
CAPTURED_TEXT = "A rare, radiant, and radiant, rare aura."


def audit(text: str) -> dict:
    tape = normalize_letters(text)
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=220)
    return {
        "text": text,
        "letters": len(tape),
        "normalized_tape": tape,
        "exact": bool(tape) and tape == tape[::-1],
        "independent_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "mechanical_checks": checks,
        "mechanically_admitted": all(checks.values()),
        "reader_eligible": False,
    }


def run() -> dict:
    model_hash = hashlib.sha256(
        subprocess.check_output(["ollama", "show", MODEL, "--modelfile"])
    ).hexdigest()
    row = audit(CAPTURED_TEXT)
    return {
        "experiment": "rhythmai-authoring-probe-20260916",
        "repair_of": "direct-constrained-authoring-20260916",
        "signature": "local-model-whole-sentence-authoring|alternate-rhythmai-model|single-scene-contract|independent-letter-audit",
        "method": "one bounded alternate-model authoring call; first output line captured verbatim after ANSI stripping",
        "model": MODEL,
        "model_modelfile_sha256": model_hash,
        "prompt": PROMPT,
        "captured_output": row,
        "exact_count": int(row["exact"]),
        "repair_action": "switch from timed-out gpt-oss authoring to a bounded RhythmAI call; next repair must return to a constructive lexical state space",
        "provenance": {
            "catalogue_used": False,
            "borrowed_text": False,
            "word_order_only": False,
            "fragments": False,
            "captured_from_local_command": True,
        },
        "reader_gate": {
            "status": "not_run",
            "reason": "short non-palindromic output failed exactness and length before any reader packet",
        },
    }


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing output: {OUT}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "exact_count": 0, "letters": 30}))
