"""Reproducible entry point for the held-out setting-frame replay.

The shared orbit-product implementation keeps the grammar and audit logic in
``two_sided_semantic_orbit_product_20260920``; this named entry point makes
the novelty ledger's setting-frame artifact distinct from the base run while
retaining the exact same deterministic implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.two_sided_semantic_orbit_product_20260920 import run


EXPERIMENT_ID = "two-sided-semantic-orbit-product-setting-frame-20260920"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs" / f"{EXPERIMENT_ID}.json",
    )
    parser.add_argument("--max-paths", type=int, default=2_000)
    parser.add_argument("--max-states", type=int, default=20_000)
    args = parser.parse_args()
    result = run(max_paths=args.max_paths, max_states=args.max_states)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
