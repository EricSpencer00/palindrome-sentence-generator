"""Preflight for a distinct route: semantic-involution frame expansion.

Instead of searching a word/phrase bank, this route starts with a short
reversible utterance and asks a human author for an *involutive* pair of
discourse frames (the outer frame is repeated in reverse lexical order).
The script is deliberately an audit harness: it prevents an attractive
rendering from being promoted until the character tape is independently
checked.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def audit(text: str) -> dict:
    tape = letters(text)
    return {"text": text, "letters": len(tape),
            "exact_letter_palindrome": tape == tape[::-1],
            "independent_reverse": tape[::-1] == tape}


def run() -> dict:
    # Authored, readable probes; these are not claimed as discoveries.
    probes = [
        "Deliver no evil. Live on, reviled.",
        "A calm deliverer: deliver no evil; live on, reviled, a calm deliverer.",
    ]
    registry = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    return {"route": "semantic-involution-frame",
            "registry_entries_read_before_run": len(registry["entries"]),
            "constructive_operator": "author semantic frame F, reversible core C, and involutive frame F^R; solve character equality during frame selection",
            "probes": [audit(p) for p in probes],
            "status": "preflight_only_no_promotion"}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
