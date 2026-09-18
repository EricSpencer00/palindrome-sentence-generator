"""Portable locations shared by the saved-output checks and rerun drivers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT if (ROOT / "inputs/brown.json.gz").is_file() else ROOT / "paper/out/evidence"


def evidence_path(name):
    path = EVIDENCE / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing evidence file: {name}. Extract evidence.zip, or run paper/build_release.py in the repository.")
    return path
