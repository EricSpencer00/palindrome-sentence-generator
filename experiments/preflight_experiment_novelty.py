"""Fail-closed preflight for a proposed construction experiment.

The registry is intentionally checked before a generator is run.  A new
filename, seed, beam width, or larger lexical bank is not enough: the
candidate signature and artifact path must be absent from both the retained
families and the explicit exclusion ledger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"


def preflight(experiment_id: str, signature: str, artifact: str) -> dict:
    data = json.loads(REGISTRY.read_text())
    retained = data.get("entries", [])
    excluded = data.get("excluded", [])
    all_rows = [("registered", row) for row in retained]
    all_rows.extend(("excluded", row) for row in excluded)

    collisions = []
    for kind, row in all_rows:
        fields = []
        if row.get("id") == experiment_id:
            fields.append("id")
        if row.get("signature") == signature:
            fields.append("signature")
        if row.get("artifact") == artifact:
            fields.append("artifact")
        if fields:
            collisions.append({"kind": kind, "id": row.get("id"), "fields": fields})
    if collisions:
        raise ValueError(f"novelty collision: {collisions}")

    artifact_path = ROOT / artifact
    if artifact_path.exists():
        raise ValueError(f"artifact already exists; choose a new path: {artifact}")
    return {
        "status": "novel",
        "experiment_id": experiment_id,
        "signature": signature,
        "artifact": artifact,
        "registered_families_checked": len(retained),
        "excluded_routes_checked": len(excluded),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, dest="experiment_id")
    parser.add_argument("--signature", required=True)
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(args.experiment_id, args.signature, args.artifact), sort_keys=True))


if __name__ == "__main__":
    main()
