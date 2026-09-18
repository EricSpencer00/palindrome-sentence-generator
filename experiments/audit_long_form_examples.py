"""Audit the frozen generated and catalogue long-form examples."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return "".join(character.lower() for character in text if character.isascii() and character.isalpha())


def pair_set(rows: list[dict]) -> set[tuple[str, str]]:
    return {(" ".join(row["left"]), " ".join(row["right"])) for row in rows}


def audit(path: Path, root: Path = ROOT) -> dict:
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    for name, expected in payload["inputs_sha256"].items():
        assert digest(root / name) == expected

    generated_pairs = pair_set(json.loads((root / "data/novel_pairs.json").read_text()))
    catalogue_pairs = pair_set(json.loads((root / "data/mirror_units.json").read_text()))
    centres = set(json.loads((root / "data/centres.json").read_text()))
    known = set(json.loads((root / "data/known_palindromes.json").read_text()))
    checks = {}
    for label, example in payload["examples"].items():
        letters = normalize(example["text"])
        assert letters == letters[::-1]
        assert example["letterPalindrome"] is True
        assert example["words"] == len(re.findall(r"[A-Za-z]+", example["text"]))
        assert example["pairs"] == len(example["units"]) == len(example["mirrors"])
        assert normalize(example["text"]) == normalize(" ".join(
            example["units"] + ([example["centre"]] if example["centre"] else [])
            + list(reversed(example["mirrors"]))))
        source_pairs = generated_pairs if label == "generated" else catalogue_pairs
        for left, right in zip(example["units"], example["mirrors"]):
            assert normalize(right) == normalize(left)[::-1]
            assert (left, right) in source_pairs
            if label == "generated":
                assert normalize(left) != normalize(left)[::-1]
                assert normalize(left + right) not in known
        if example["centre"]:
            assert example["centre"] in centres
        checks[label] = {
            "words": example["words"],
            "letters": len(letters),
            "pairs": example["pairs"],
            "exact_palindrome": True,
            "source": example["source"],
            "borrowed": example["borrowed"],
        }
    assert checks["generated"] == {
        "words": 101, "letters": 342, "pairs": 24,
        "exact_palindrome": True, "source": "generated", "borrowed": False,
    }
    assert checks["catalogue"] == {
        "words": 72, "letters": 237, "pairs": 9,
        "exact_palindrome": True, "source": "catalogue", "borrowed": True,
    }
    return {"status": "pass", "examples": checks, "result_sha256": digest(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", nargs="?", default="runs/long-form-examples-2026-09-11/examples.json")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit(Path(args.result), Path(args.root))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
