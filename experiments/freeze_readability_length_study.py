"""Freeze a blinded human study of readability versus exact-palindrome length.

The study does not use an LLM score to select examples.  It renders fixed,
seeded v3 compositions at three length targets, adds matched ordinary-English
and shuffled-word controls, and keeps the source key separate from the packet
given to readers.  Human ratings are intentionally blank at freeze time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.mirror_cost import load_corpus, sentences
from llm_palindrome.validator import is_palindrome, normalize


BANDS = (("short", 80), ("medium", 240), ("long", 480))
RUBRIC = {
    "grammaticality": "0 = word salad; 1 = fragments dominate; 2 = mostly grammatical but strained; 3 = consistently grammatical English.",
    "subject": "0 = no identifiable subject or intent; 1 = hints of one; 2 = a recoverable subject or intent; 3 = a clear, specific subject or intent.",
    "coherence": "0 = no connected meaning; 1 = local phrases only; 2 = a recoverable connected meaning; 3 = clear coherent prose.",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", text)


def build_real_span(pool: list[str], target_letters: int, rng: random.Random) -> str:
    """Start at a sentence boundary and stop at the nearest word boundary."""
    for _ in range(len(pool) * 2):
        start = rng.randrange(len(pool))
        selected: list[str] = []
        letters = 0
        for source in pool[start:]:
            for token in words(source):
                if selected and letters + len(token) > target_letters:
                    return " ".join(selected) + "."
                selected.append(token)
                letters += len(token)
                if letters >= target_letters:
                    return " ".join(selected) + "."
    raise RuntimeError("Could not produce a length-matched prose control")


def system_item(band: str, target_letters: int, seed: int) -> dict:
    raise RuntimeError(
        "legacy readability-length study is retired: it draws prohibited v3 and catalogue material"
    )
    from server import v3

    row = v3.composition(seed=seed, letters=target_letters, chops=None,
                         longest_first=False, centre=None, novel=True,
                         hierarchical=True)
    text = row["text"]
    if not is_palindrome(text):
        raise AssertionError("Rendered system output must preserve exactness")
    return {
        "source": "system",
        "band": band,
        "system_seed": seed,
        "target_letters": target_letters,
        "letters": row["letters"],
        "words": row["words"],
        "pairs": row["pairs"],
        "hierarchical": row["hierarchical"],
        "text": text,
    }


def catalogue_items(count: int, rng: random.Random) -> list[dict]:
    rows = json.loads(Path("data/v3_bank.json").read_text())
    pool = [row for row in rows if row.get("source") == "catalogue"
            and 20 <= len(normalize(row["text"])) <= 80]
    rng.shuffle(pool)
    if len(pool) < count:
        raise RuntimeError("Not enough catalogue references")
    out = []
    for row in pool[:count]:
        text = row["text"]
        if not is_palindrome(text):
            raise AssertionError("Catalogue reference is not an exact palindrome")
        out.append({
            "source": "catalogue_reference",
            "band": "reference",
            "letters": len(normalize(text)),
            "words": len(words(text)),
            "text": text,
        })
    return out


def study_rows(per_band: int, seed: int) -> tuple[list[dict], dict]:
    raise RuntimeError(
        "legacy readability-length study is retired: it draws prohibited v3 and catalogue material"
    )
    rng = random.Random(seed)
    corpus, corpus_meta = load_corpus()
    sentence_pool = sentences(corpus)
    rows: list[dict] = []
    for band_index, (band, target_letters) in enumerate(BANDS):
        for item_index in range(per_band):
            generated = system_item(band, target_letters,
                                    band_index * 10_000 + item_index)
            rows.append(generated)
            prose = build_real_span(sentence_pool, generated["letters"], rng)
            prose_words = words(prose)
            salad_words = list(prose_words)
            rng.shuffle(salad_words)
            rows.extend((
                {
                    "source": "real_prose_control",
                    "band": band,
                    "letters": len(normalize(prose)),
                    "words": len(prose_words),
                    "text": prose,
                },
                {
                    "source": "shuffled_word_control",
                    "band": band,
                    "letters": len(normalize(" ".join(salad_words))),
                    "words": len(salad_words),
                    "text": " ".join(salad_words) + ".",
                },
            ))
    rows.extend(catalogue_items(per_band, rng))
    rng.shuffle(rows)
    for index, row in enumerate(rows, 1):
        row["id"] = f"R{index:03d}"
    return rows, corpus_meta


def instructions(item_count: int) -> str:
    dimensions = "\n".join(f"- **{name}:** {text}" for name, text in RUBRIC.items())
    return f"""# Blinded readability study

You will read {item_count} short passages. Some are exact palindromes; some are
ordinary English or shuffled-word controls. Their source is intentionally not
disclosed. Please read each passage independently and do not search for it.

For every item, assign one integer from 0 to 3 on each dimension:

{dimensions}

Do not reward or penalize a passage merely for being long, short, unusual,
palindromic, archaic, or poetic. Rate what a reader can understand. If you
cannot fairly judge an item, leave its three scores blank and explain why in
`notes`. Do not discuss items with other raters before submitting the CSV.
"""


def write_csv_template(path: Path, ids: Iterable[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "rater_id", "item_id", "grammaticality", "subject", "coherence", "notes",
        ])
        writer.writeheader()
        for item_id in ids:
            writer.writerow({"rater_id": "", "item_id": item_id,
                             "grammaticality": "", "subject": "",
                             "coherence": "", "notes": ""})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-band", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    raise RuntimeError(
        "legacy readability-length study building is retired: it has no admissible candidate material"
    )
    if args.per_band <= 0:
        parser.error("--per-band must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=False)

    rows, corpus_meta = study_rows(args.per_band, args.seed)
    blind = [{"id": row["id"], "text": row["text"]} for row in rows]
    key = [{key: value for key, value in row.items() if key != "text"}
           for row in rows]
    protocol = {
        "question": "Do novel exact palindromes from the fixed v3 composer retain readability as requested length increases?",
        "status": "frozen_pending_independent_human_ratings",
        "bands": [{"name": name, "target_letters": letters} for name, letters in BANDS],
        "system": {
            "endpoint": "server.v3.composition",
            "novel": True,
            "hierarchical": True,
            "longest_first": False,
            "system_seeds": "10,000 * band_index + item_index",
            "selection": "fixed before human ratings; no language-quality score selects items",
        },
        "controls": ["real_prose_control", "shuffled_word_control", "catalogue_reference"],
        "per_band": args.per_band,
        "items": len(rows),
        "required_independent_raters": 3,
        "rubric": RUBRIC,
        "analysis": {
            "primary": "mean coherence by source and system length band, with all individual ratings retained",
            "secondary": "grammaticality and identifiable subject/intent by source and band",
            "reliability": "ordinal Krippendorff alpha per dimension when at least three raters return scores",
            "guardrail": "No language-quality claim is made before independent ratings and control separation are available.",
        },
        "corpus": corpus_meta,
        "input_sha256": {
            "data/v3_bank.json": sha256(Path("data/v3_bank.json")),
            "data/lexicon.txt": sha256(Path("data/lexicon.txt")),
        },
        "source_sha256": {
            "experiments/freeze_readability_length_study.py": sha256(Path(__file__)),
            "server/v3.py": sha256(Path("server/v3.py")),
        },
    }
    (args.out_dir / "blind-items.json").write_text(json.dumps(blind, indent=2) + "\n")
    (args.out_dir / "key.json").write_text(json.dumps(key, indent=2) + "\n")
    (args.out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (args.out_dir / "HUMAN-INSTRUCTIONS.md").write_text(instructions(len(rows)))
    write_csv_template(args.out_dir / "human-ratings.csv", (row["id"] for row in rows))
    manifest = {path.name: sha256(path) for path in args.out_dir.iterdir() if path.is_file()}
    (args.out_dir / "MANIFEST-SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "items": len(rows),
                      "system_items": sum(row["source"] == "system" for row in rows),
                      "controls": sum(row["source"] != "system" for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
