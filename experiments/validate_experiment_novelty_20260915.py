"""Validate the experiment novelty registry against the checked-out tree."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"


def validate(path: Path = REGISTRY) -> dict:
    data = json.loads(path.read_text())
    entries = data["entries"]
    excluded = data.get("excluded", [])
    ids = [row["id"] for row in entries]
    signatures = [row["signature"] for row in entries]
    artifacts = [row["artifact"] for row in entries]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate experiment id")
    if len(signatures) != len(set(signatures)):
        raise AssertionError("duplicate state-space signature")
    if len(artifacts) != len(set(artifacts)):
        raise AssertionError("one artifact is being counted as multiple experiments")
    missing = [row["artifact"] for row in entries if not (ROOT / row["artifact"]).exists()]
    if missing:
        raise AssertionError(f"missing artifacts: {missing}")
    excluded_ids = [row["id"] for row in excluded]
    excluded_signatures = [row["signature"] for row in excluded]
    excluded_artifacts = [row["artifact"] for row in excluded]
    if len(excluded_ids) != len(set(excluded_ids)):
        raise AssertionError("duplicate excluded experiment id")
    if set(excluded_ids) & set(ids):
        raise AssertionError("excluded experiment is registered")
    if set(excluded_signatures) & set(signatures):
        raise AssertionError("excluded signature is registered")
    if set(excluded_artifacts) & set(artifacts):
        raise AssertionError("excluded artifact is registered")
    missing_excluded = [row["artifact"] for row in excluded if not (ROOT / row["artifact"]).exists()]
    if missing_excluded:
        raise AssertionError(f"missing excluded artifacts: {missing_excluded}")
    return {"entries": len(entries), "unique_signatures": len(set(signatures)), "unique_artifacts": len(set(artifacts)), "excluded": len(excluded), "missing": missing + missing_excluded}


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
