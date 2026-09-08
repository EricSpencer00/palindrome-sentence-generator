"""Held-out evaluation and Pareto selection for diversity-debug artifacts."""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_palindrome.lm_scoring import GPT2Scorer
from llm_palindrome.pareto import pareto_front
from llm_palindrome.wordorder import shuffles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--shuffles", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for path in glob.glob(str(args.run_dir / "summary_r*.json")):
        rows.extend(json.load(open(path))["rows"])
    # Repeated seeds are not independent evidence. Judge each distinct output
    # once, while preserving its generating arm.
    unique = {}
    for row in rows:
        if row["closed"]:
            unique.setdefault((row["arm"], row["text"]), dict(row))
    candidates = list(unique.values())

    texts = []
    spans = []
    for index, row in enumerate(candidates):
        words = row["text"].split()
        variants = [words] + shuffles(words, args.shuffles, seed=index)
        start = len(texts)
        texts.extend(" ".join(variant) for variant in variants)
        spans.append((start, len(texts)))

    scores = GPT2Scorer(args.model).score_details(texts)
    for row, (start, end) in zip(candidates, spans):
        per_token = [score["per_token"] for score in scores[start:end]]
        row["gpt2_per_token"] = per_token[0]
        row["gpt2_shuffle_gain"] = (per_token[0] - statistics.mean(per_token[1:])
                                      if len(per_token) > 1 else 0.0)

    front = pareto_front(candidates, (
        "gpt2_shuffle_gain", "distinct_ratio", "attested_pair_rate"))
    front_ids = {(row["arm"], row["text"]) for row in front}
    for row in candidates:
        row["pareto"] = (row["arm"], row["text"]) in front_ids

    arms = {}
    for name in ("baseline", "diverse", "diverse_sem"):
        arm = [row for row in candidates if row["arm"] == name]
        arms[name] = {
            "unique": len(arm),
            "mean_gpt2_per_token": statistics.mean(
                row["gpt2_per_token"] for row in arm),
            "mean_gpt2_shuffle_gain": statistics.mean(
                row["gpt2_shuffle_gain"] for row in arm),
            "best_gpt2_shuffle_gain": max(
                row["gpt2_shuffle_gain"] for row in arm),
            "pareto": sum(row["pareto"] for row in arm),
        }
    result = {"config": {"model": args.model, "shuffles": args.shuffles},
              "arms": arms, "pareto": front, "candidates": candidates}
    rendered = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(rendered + "\n")
    print(json.dumps({"config": result["config"], "arms": arms,
                      "pareto_count": len(front)}, indent=2))


if __name__ == "__main__":
    main()
