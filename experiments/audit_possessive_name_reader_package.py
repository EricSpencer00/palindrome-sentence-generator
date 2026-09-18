"""Diagnose, but never certify, the possessive-name reader materials.

The report is intentionally separate from the generator.  It records a few
auditable surface properties for each blinded item: local Brown-bigram order
gain against a deterministic shuffle of its own words, shipped-lexicon
coverage, and repetition.  The candidate/shuffle pair preserves the exact word
multiset.  These measurements can flag a broken control or an obvious word
salad; they cannot determine whether a person finds a sentence grammatical or
understandable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.audit_programmatic_readability import BrownBigramModel, order_gain
from llm_palindrome.lexicon import load_lexicon

WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def tokens(text: str) -> list[str]:
    """Keep possessives intact; punctuation never changes the letter tape."""
    return [word.casefold().replace("'", "") for word in WORD.findall(text)]


def load_blinded_items(package: Path) -> tuple[list[dict], list[dict]]:
    """Recover opaque item texts from one rater form and conditions internally."""
    key = json.loads((package / "internal" / "condition-key.json").read_text())
    form = json.loads((package / "rater-package" / "R001.json").read_text())
    texts = {row["opaque_id"]: row["text"] for row in form["items"]}
    if len(texts) != len(form["items"]):
        raise ValueError("opaque IDs are not unique in the frozen rater form")
    if {row["opaque_id"] for row in key} != set(texts):
        raise ValueError("condition key and blinded rater form disagree")
    return key, [{"opaque_id": row["opaque_id"], "text": texts[row["opaque_id"]]}
                 for row in key]


def validate_controls(key: list[dict], blinded: list[dict]) -> None:
    by_id = {row["opaque_id"]: row["text"] for row in blinded}
    by_block: dict[str, dict[str, str]] = {}
    for row in key:
        by_block.setdefault(row["block"], {})[row["condition"]] = row["opaque_id"]
    for block, conditions in by_block.items():
        if set(conditions) != {"candidate", "intact_prose", "word_shuffle"}:
            raise ValueError(f"{block} lacks a required blinded condition")
        candidate = tokens(by_id[conditions["candidate"]])
        shuffled = tokens(by_id[conditions["word_shuffle"]])
        if sorted(candidate) != sorted(shuffled) or candidate == shuffled:
            raise ValueError(f"{block} shuffle does not preserve and reorder candidate words")


def item_report(row: dict, text: str, model: BrownBigramModel, lexicon: frozenset[str],
                *, seed: int, shuffles: int) -> dict:
    words = tokens(text)
    observed, gain = order_gain(model, words, row["opaque_id"], seed, shuffles)
    bigrams = list(zip(words, words[1:]))
    return {
        "opaque_id": row["opaque_id"],
        "block": row["block"],
        "condition": row["condition"],
        "rendered": text,
        "normalized_letters": len("".join(words)),
        "word_count": len(words),
        "unique_word_count": len(set(words)),
        "shipped_lexicon_coverage": sum(word in lexicon for word in words) / len(words),
        "repeated_bigram_count": len(bigrams) - len(set(bigrams)),
        "brown_bigram_logprob": observed,
        "brown_order_gain_vs_own_word_shuffles": gain,
    }


def audit(package: Path, *, seed: int = 2026091203, shuffles: int = 128) -> dict:
    key, blinded = load_blinded_items(package)
    validate_controls(key, blinded)
    texts = {row["opaque_id"]: row["text"] for row in blinded}
    model = BrownBigramModel.from_brown()
    lexicon = load_lexicon(str(ROOT / "data" / "lexicon.txt"))
    rows = [item_report(row, texts[row["opaque_id"]], model, lexicon,
                        seed=seed, shuffles=shuffles) for row in key]
    by_block: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_block.setdefault(row["block"], {})[row["condition"]] = row
    contrasts = []
    intact_gain_positive = []
    for block, conditions in sorted(by_block.items()):
        candidate, shuffle = conditions["candidate"], conditions["word_shuffle"]
        intact_gain_positive.append(
            conditions["intact_prose"]["brown_order_gain_vs_own_word_shuffles"] > 0
        )
        contrasts.append({
            "block": block,
            "candidate_minus_word_shuffle_brown_logprob": (
                candidate["brown_bigram_logprob"] - shuffle["brown_bigram_logprob"]
            ),
            "candidate_minus_word_shuffle_order_gain": (
                candidate["brown_order_gain_vs_own_word_shuffles"]
                - shuffle["brown_order_gain_vs_own_word_shuffles"]
            ),
        })
    calibrated = all(intact_gain_positive)
    return {
        "status": "diagnostic_only_not_a_human_readability_result",
        "package": str(package),
        "method": {
            "brown_bigram": "add-alpha word bigram model trained on NLTK Brown",
            "order_gain": "observed mean bigram log probability minus deterministic own-word shuffles",
            "shuffle_count": shuffles,
            "random_seed": seed,
        },
        "hard_control_checks": {
            "candidate_and_shuffle_have_identical_word_multisets": True,
            "candidate_and_shuffle_have_different_orders": True,
        },
        "local_order_metric_calibration": {
            "test": "every intact-prose control must have positive gain over its own deterministic word shuffles",
            "per_block_pass": intact_gain_positive,
            "usable_for_this_package": calibrated,
            "decision": (
                "The metric is a local-order diagnostic only; no scalar is ever a readability result."
                if calibrated else
                "The metric does not calibrate on these proper-name controls and is therefore unusable "
                "for ranking or filtering this package. Human ratings remain the only readability gate."
            ),
        },
        "limits": [
            "The Brown score measures only local word order and is not a grammar, meaning, or readability score.",
            "Lexicon coverage treats proper names conservatively and cannot distinguish a coherent event from a word list.",
            "No value in this report selects candidates, establishes readability, or substitutes for blinded human ratings.",
        ],
        "items": rows,
        "candidate_vs_word_shuffle_contrasts": contrasts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026091203)
    parser.add_argument("--shuffles", type=int, default=128)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    if args.shuffles < 2:
        parser.error("--shuffles must be at least 2")
    result = audit(args.package, seed=args.seed, shuffles=args.shuffles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "items": len(result["items"]),
                      "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
