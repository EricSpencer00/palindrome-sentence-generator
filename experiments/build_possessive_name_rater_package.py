"""Retired historical builder for the rejected catalogue-family ablation.

It is deliberately non-operational. Its prior within-rater matched-variant
design and condition-correlated IDs are not suitable for a blinded reader
study. New reader studies must use a separately reviewed builder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re


RATER_COUNT = 24
BASE_SEED = 2026091202


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)


def shuffled(text: str, *, seed: int) -> str:
    units = tokens(text)
    random.Random(seed).shuffle(units)
    candidate = " ".join(units) + "."
    if candidate.casefold() == text.casefold():
        raise AssertionError("shuffle did not change the word order")
    return candidate


def natural_control(candidate: str) -> str:
    substitutions = {
        "Marge lets Hara see Sarah's telegram.": "Marge lets Hara read Sarah's telegram.",
        "Marge lets Aino see Sonia's telegram.": "Marge lets Aino read Sonia's telegram.",
    }
    try:
        return substitutions[candidate]
    except KeyError as exc:
        raise ValueError(f"no intact control specified for {candidate!r}") from exc


def item(block: str, condition: str, text: str, opaque_id: str) -> dict:
    return {
        "block": block,
        "condition": condition,
        "opaque_id": opaque_id,
        "text": text,
        "word_multiset": sorted(token.casefold() for token in tokens(text)),
    }


def build(source: Path) -> tuple[list[dict], list[dict]]:
    del source
    raise RuntimeError(
        "retired: this builder cannot package catalogue-family material or any new reader study"
    )


def write_package(out_dir: Path, source: Path) -> dict:
    items, key = build(source)
    internal, raters = out_dir / "internal", out_dir / "rater-package"
    internal.mkdir(parents=True)
    raters.mkdir()
    (internal / "condition-key.json").write_text(json.dumps(key, indent=2) + "\n")
    instructions = {
        "title": "Short-utterance reading study",
        "instructions": (
            "Read each item as ordinary English. Do not try to detect patterns or infer how the "
            "items were made. Answer from your first reading, without external search or editing."
        ),
        "questions": [
            {"id": "grammar", "prompt": "How grammatically complete is this utterance?", "scale": [1, 5]},
            {"id": "meaning", "prompt": "How coherent is its meaning?", "scale": [1, 5]},
            {"id": "paraphrase", "prompt": "In your own words, who does what?", "type": "free_text"},
            {"id": "repair", "prompt": "Would you change anything to make it clearer? If so, what?", "type": "free_text"},
        ],
    }
    for rater_number in range(1, RATER_COUNT + 1):
        order = list(items)
        random.Random(BASE_SEED + 1000 + rater_number).shuffle(order)
        blind_items = [{"opaque_id": row["opaque_id"], "text": row["text"]}
                       for row in order]
        payload = instructions | {"rater_id": f"R{rater_number:03d}", "items": blind_items}
        (raters / f"R{rater_number:03d}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (raters / "README.md").write_text(
        "# Reader materials\n\n"
        "Complete one JSON form in the supplied order. Each item is an ordinary short utterance. "
        "Do not search for it or discuss items with other raters.\n"
    )
    (internal / "analysis-plan.md").write_text(
        "# Frozen analysis plan\n\n"
        "Primary endpoint: per-item grammatical-completeness and meaning-coherence ratings, reported "
        "with every individual response, median, and bootstrap confidence interval. The candidate is "
        "not considered reader-readable unless a pre-specified majority rates both dimensions at least "
        "4 and free-text paraphrases recover the intended permission event without prompted repair.\n\n"
        "Secondary endpoint: compare each candidate with its intact prose and word-shuffle controls in "
        "a mixed-effects ordinal model with rater and item random effects, after at least 24 completed "
        "independent rater forms. Do not collapse the two candidate items into a general population claim.\n"
    )
    manifest = {str(path.relative_to(out_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(out_dir.rglob("*")) if path.is_file()}
    (out_dir / "MANIFEST-SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"items": len(items), "raters": RATER_COUNT, "manifest": manifest,
            "status": "historical_builder_output_not_a_validated_reader_study"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    result = write_package(args.out_dir, args.source)
    print(json.dumps({"out_dir": str(args.out_dir), "items": result["items"],
                      "raters": result["raters"]}, indent=2))


if __name__ == "__main__":
    main()
