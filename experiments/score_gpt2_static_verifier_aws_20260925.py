"""Run the frozen GPT-2 scoring code and label its AWS CPU execution truthfully."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.score_gpt2_static_verifier_20260925 import score as frozen_score


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(input_path: Path, output_path: Path, revision: str, batch_size: int,
        execution_host: str) -> dict:
    result = frozen_score(input_path, output_path, revision, batch_size)
    result["model"]["device"] = "CPU"
    result["model"]["execution_host"] = execution_host
    result["model"]["base_scorer_sha256"] = sha256_file(
        Path(__file__).with_name("score_gpt2_static_verifier_20260925.py")
    )
    result["model"]["execution_wrapper_sha256"] = sha256_file(Path(__file__))
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--execution-host", required=True)
    args = parser.parse_args()
    result = run(args.input, args.output, args.revision, args.batch_size, args.execution_host)
    print(json.dumps({
        "output": str(args.output),
        "items": len(result["scores"]),
        "revision": result["model"]["revision"],
        "execution_host": result["model"]["execution_host"],
    }, indent=2))


if __name__ == "__main__":
    main()
