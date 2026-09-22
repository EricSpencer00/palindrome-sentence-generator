"""Build a frozen blinded reader package for the 54-letter NP candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re


RATER_COUNT = 24
BASE_SEED = 2026092207


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)


def shuffled(text: str, *, seed: int) -> str:
    original = tokens(text)
    words = list(original)
    random.Random(seed).shuffle(words)
    changed = " ".join(words) + "."
    if [word.casefold() for word in tokens(changed)] == [
            word.casefold() for word in original]:
        raise AssertionError("shuffle did not change order")
    return changed


def _source_candidate(source: Path) -> str:
    payload = json.loads(source.read_text())
    rows = payload.get("reader_study_candidates", [])
    if len(rows) != 1:
        raise ValueError("source must contain exactly one reader-study candidate")
    row = rows[0]
    audit = row["independent_exact_audit"]
    if not (row["mechanically_admitted"] and audit["two_pointer_exact"]
            and audit["project_validator"] and audit["hashes_agree"]):
        raise ValueError("source candidate has not passed every mechanical gate")
    if row.get("human_reader_status") != "not_run":
        raise ValueError("source already contains reader results")
    return row["rendered"]


def build(source: Path) -> tuple[list[dict], list[dict]]:
    candidate = _source_candidate(source)
    incumbent = "An aide rips nine memos; some men inspire Diana."
    intact = (
        "An aide tears up nine memos about an office hero. "
        "Several additional local men inspire Diana."
    )
    surfaces = [
        ("candidate_54", candidate),
        ("exact_baseline_38", incumbent),
        ("intact_control", intact),
        ("candidate_shuffle", shuffled(candidate, seed=BASE_SEED + 1)),
        ("intact_shuffle", shuffled(intact, seed=BASE_SEED + 2)),
    ]
    opaque_ids = ["item-k7v2", "item-r4m9", "item-c8q1", "item-w3h6", "item-p9x5"]
    items, key = [], []
    for opaque_id, (condition, text) in zip(opaque_ids, surfaces):
        items.append({"opaque_id": opaque_id, "text": text})
        key.append({
            "opaque_id": opaque_id,
            "condition": condition,
            "text": text,
            "word_multiset": sorted(word.casefold() for word in tokens(text)),
        })
    return items, key


def write_package(out_dir: Path, source: Path) -> dict:
    items, key = build(source)
    internal = out_dir / "internal"
    raters = out_dir / "rater-package"
    internal.mkdir(parents=True)
    raters.mkdir()
    (internal / "condition-key.json").write_text(json.dumps(key, indent=2) + "\n")
    (internal / "source-sha256.txt").write_text(
        hashlib.sha256(source.read_bytes()).hexdigest() + "  " + source.name + "\n"
    )
    instructions = {
        "title": "Short English passage reading study",
        "instructions": (
            "Read each item as ordinary English. Do not search for it, try to "
            "detect a pattern, or discuss it with another rater. Judge your "
            "first reading; the study does not test spelling puzzles."
        ),
        "questions": [
            {"id": "grammar", "prompt": "How grammatically complete is this passage?", "scale": [1, 5]},
            {"id": "meaning", "prompt": "How coherent and understandable is its meaning?", "scale": [1, 5]},
            {"id": "paraphrase", "prompt": "In your own words, who does what?", "type": "free_text"},
            {"id": "repair", "prompt": "Would you change anything to make it clearer? If so, what?", "type": "free_text"},
        ],
    }
    for number in range(1, RATER_COUNT + 1):
        order = list(items)
        random.Random(BASE_SEED + 1000 + number).shuffle(order)
        payload = instructions | {
            "rater_id": f"R{number:03d}",
            "items": order,
            "responses": [],
        }
        (raters / f"R{number:03d}.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
    (raters / "README.md").write_text(
        "# Reader materials\n\n"
        "Complete exactly one assigned JSON form in its supplied order. "
        "Raters must not see the `internal` directory or know which items are palindromes.\n"
    )
    (internal / "analysis-plan.md").write_text(
        "# Frozen analysis plan\n\n"
        "The 54-letter candidate succeeds only if at least 16 of 24 independent "
        "raters assign both grammar and meaning scores of 4 or 5, and at least "
        "16 unprompted paraphrases recover both events: an aide destroys memos "
        "and men inspire Diana. Report every individual response, medians, and "
        "bootstrap 95% confidence intervals. Compare candidate, exact baseline, "
        "intact control, and both shuffled controls only after all 24 forms are "
        "frozen. Exclude no response based on its score; record only predeclared "
        "duplicate/incomplete-form exclusions. Programmatic scores cannot replace "
        "these outcomes.\n"
    )
    manifest = {
        str(path.relative_to(out_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out_dir.rglob("*")) if path.is_file()
    }
    (out_dir / "MANIFEST-SHA256.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return {
        "items": len(items), "raters": RATER_COUNT,
        "manifest_entries": len(manifest),
        "status": "frozen_unrun_blinded_reader_package",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    result = write_package(args.out_dir, args.source)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
