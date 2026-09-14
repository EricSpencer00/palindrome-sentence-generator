"""Evaluate Polaris extension and control artifacts without certifying readability.

The exact tape check is independent of the search driver. Brown order gain is
paired with deterministic own-word shuffles, while familiarity and repetition
remain descriptive. The report keeps the fixed human-selected center separate
from the free-search controls so a length increase cannot masquerade as prose
quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.validator import is_palindrome, normalize
from experiments.audit_programmatic_readability import (
    BrownBigramModel, order_gain, repeated_bigram_rate, safe_mean, tokens,
)
from wordfreq import zipf_frequency


CENTER = "an aide rips nine memos some men inspire diana"


def exact_row(text: str, source: str, arm: str, item_id: str,
              model: BrownBigramModel, seed: int, shuffles: int) -> dict:
    words = tokens(text)
    observed, gain = order_gain(model, words, item_id, seed, shuffles)
    return {
        "id": item_id,
        "source": source,
        "arm": arm,
        "text": text,
        "letters": len(normalize(text)),
        "words": len(words),
        "independent_exact": bool(normalize(text)) and normalize(text) == normalize(text)[::-1],
        "validator_exact": is_palindrome(text),
        "brown_bigram_logprob": observed,
        "brown_order_gain_vs_own_shuffle": gain,
        "mean_zipf_frequency": safe_mean(zipf_frequency(word, "en") for word in words),
        "repeated_word_rate": 1 - len(set(words)) / len(words) if words else 0.0,
        "repeated_bigram_rate": repeated_bigram_rate(words),
    }


def load_center(root: Path, model: BrownBigramModel, seed: int, shuffles: int):
    rows = []
    for arm in ("score", "letters"):
        for path in sorted(root.joinpath(arm).glob("summary_*.json")):
            payload = json.loads(path.read_text())
            for row in payload["rows"]:
                rows.append(exact_row(row["text"], "fixed_center_extension", arm,
                                      f"{arm}-{row['rank']}-{row['seed']}",
                                      model, seed, shuffles))
    return rows


def load_sentence_bank(root: Path, model: BrownBigramModel, seed: int, shuffles: int):
    rows = []
    for path in sorted(root.glob("summary_*.json")):
        payload = json.loads(path.read_text())
        for row in payload["rows"]:
            if row.get("text"):
                rows.append(exact_row(row["text"], "free_sentence_bank", "free",
                                      f"free-{row['rank']}-{row['seed']}",
                                      model, seed, shuffles))
    return rows


def load_grammar(root: Path, model: BrownBigramModel, seed: int, shuffles: int):
    rows = []
    for path in sorted(root.glob("summary_*.json")):
        payload = json.loads(path.read_text())
        for index, row in enumerate(payload.get("rows", [])):
            rows.append(exact_row(row["text"], "grammar_extension", "grammar",
                                  f"grammar-{index}",
                                  model, seed, shuffles))
    return rows


def summary(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    closed = [row for row in rows if row["independent_exact"]]
    return {
        "n": len(rows),
        "exact": len(closed),
        "length_min": min(row["letters"] for row in rows),
        "length_mean": statistics.fmean(row["letters"] for row in rows),
        "length_max": max(row["letters"] for row in rows),
        "mean_zipf": statistics.fmean(row["mean_zipf_frequency"] for row in rows),
        "mean_brown_order_gain": statistics.fmean(
            row["brown_order_gain_vs_own_shuffle"] for row in rows
            if row["brown_order_gain_vs_own_shuffle"] is not None),
        "mean_repeated_word_rate": statistics.fmean(row["repeated_word_rate"] for row in rows),
        "mean_repeated_bigram_rate": statistics.fmean(row["repeated_bigram_rate"] for row in rows),
    }


def audit(center_root: Path, sentence_root: Path | None,
          grammar_root: Path | None, *, seed: int, shuffles: int) -> dict:
    model = BrownBigramModel.from_brown()
    center_rows = load_center(center_root, model, seed, shuffles)
    free_rows = load_sentence_bank(sentence_root, model, seed, shuffles) if sentence_root else []
    grammar_rows = load_grammar(grammar_root, model, seed, shuffles) if grammar_root else []
    baseline = exact_row(CENTER, "human_selected_seed", "seed", "center-seed",
                        model, seed, shuffles)
    groups = {}
    for arm in ("score", "letters"):
        groups[f"fixed_center_extension/{arm}"] = summary(
            [row for row in center_rows if row["arm"] == arm])
    groups["free_sentence_bank/free"] = summary(free_rows)
    groups["human_selected_seed/seed"] = summary([baseline])
    groups["grammar_extension/grammar"] = summary(grammar_rows)
    center_rows.sort(key=lambda row: (-row["letters"], row["id"]))
    coherent = sorted(
        center_rows,
        key=lambda row: (-(row["brown_order_gain_vs_own_shuffle"] or float("-inf")),
                         -row["letters"], row["id"]),
    )
    return {
        "status": "diagnostic_not_human_readability_result",
        "center": CENTER,
        "center_letters": len(normalize(CENTER)),
        "center_sha256": hashlib.sha256(CENTER.encode()).hexdigest(),
        "method": {
            "brown_bigram": "add-alpha NLTK Brown word bigram",
            "order_gain": "observed mean log probability minus deterministic own-word shuffles",
            "shuffle_count": shuffles,
            "random_seed": seed,
        },
        "limits": [
            "Exactness is mechanically audited; no programmatic feature certifies human readability.",
            "Brown order gain measures local word order only and cannot establish discourse coherence.",
            "The free sentence-bank arm is a control, not a source of generated prose claims.",
        ],
        "groups": groups,
        "top_longest_fixed_center": center_rows[:8],
        "top_order_gain_fixed_center": coherent[:8],
        "grammar_extension_rows": grammar_rows,
        "human_selected_seed": baseline,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("center_root", type=Path)
    ap.add_argument("--sentence-root", type=Path)
    ap.add_argument("--grammar-root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--shuffles", type=int, default=32)
    args = ap.parse_args()
    report = audit(args.center_root, args.sentence_root, args.grammar_root,
                   seed=args.seed, shuffles=args.shuffles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "groups": report["groups"]}, indent=2))


if __name__ == "__main__":
    main()
