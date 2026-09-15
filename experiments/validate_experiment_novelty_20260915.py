"""Validate the experiment novelty registry against the checked-out tree."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"


def validate(path: Path = REGISTRY) -> dict:
    data = json.loads(path.read_text())
    entries = data["entries"]
    ids = [row["id"] for row in entries]
    signatures = [row["signature"] for row in entries]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate experiment id")
    if len(signatures) != len(set(signatures)):
        raise AssertionError("duplicate state-space signature")
    missing = [row["artifact"] for row in entries if not (ROOT / row["artifact"]).exists()]
    if missing:
        raise AssertionError(f"missing artifacts: {missing}")
    return {"entries": len(entries), "unique_signatures": len(set(signatures)), "missing": missing}


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
