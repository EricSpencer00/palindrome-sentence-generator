"""Audit the frozen mirror-cost result without loading a language model.

This checker verifies the experimental design, saved spans, vocabulary-derived
coverage values, and arithmetic relationships in the reported rows.  It cannot
independently reproduce model logits; rerunning ``experiments/mirror_cost.py``
is required for that stronger check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments.mirror_cost import coverage, load_vocab, segment
from llm_palindrome.validator import normalize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "runs/mirror-cost-2026-09-11/results.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{actual!r} != {expected!r}")


def audit(path: Path) -> dict:
    result = json.loads(path.read_text())
    assert result["schema_version"] == 2
    design = result["design"]
    models = [row["name"] for row in result["models"]]
    strategies = design["segmentation_strategies"]
    lengths = design["target_lengths"]
    expected_cells = {(model, strategy, length)
                      for model in models
                      for strategy in strategies
                      for length in lengths}
    rows = result["rows"]
    observed_cells = {(row["model"], row["strategy"], row["n_letters"])
                      for row in rows}
    assert observed_cells == expected_cells
    assert len(rows) == len(expected_cells)

    vocab_path = ROOT / result["vocabulary"]["path"]
    assert digest(vocab_path) == result["vocabulary"]["sha256"]
    vocab = load_vocab(str(vocab_path))
    assert len(vocab) == result["vocabulary"]["filtered_entries"]

    saved_samples = result["samples"]
    coverage_by_cell = {}
    for length in lengths:
        sample = saved_samples[str(length)]
        assert len(sample) == design["spans_per_length"]
        for item in sample:
            assert normalize(item["natural"]) == item["letters"]
            assert len(item["letters"]) >= length
        for strategy in strategies:
            forward = []
            reverse = []
            for item in sample:
                letters = item["letters"]
                seg_forward = segment(letters, vocab, strategy)
                seg_reverse = segment(letters[::-1], vocab, strategy)
                assert "".join(seg_forward) == letters
                assert "".join(seg_reverse) == letters[::-1]
                forward.append(coverage(seg_forward, vocab))
                reverse.append(coverage(seg_reverse, vocab))
            coverage_by_cell[(strategy, length)] = (
                sum(forward) / len(forward), sum(reverse) / len(reverse))

    for row in rows:
        assert row["n_spans"] == design["spans_per_length"]
        assert row["n_scored"] == design["spans_per_length"]
        close(row["mirror_cost"], row["bits_reversed"] - row["bits_forward"])
        close(row["mirror_cost_se"],
              row["mirror_cost_sd"] / math.sqrt(row["n_scored"]))
        close(row["thinning_per_letter"], 2.0 ** row["mirror_cost"])
        expected_forward, expected_reverse = coverage_by_cell[
            (row["strategy"], row["n_letters"])]
        close(row["coverage_forward"], expected_forward)
        close(row["coverage_reversed"], expected_reverse)

    costs = [row["mirror_cost"] for row in rows]
    resegmentation = [row["bits_forward"] - row["bits_natural"] for row in rows]
    by_model = {}
    for model in models:
        selected = [row for row in rows if row["model"] == model]
        by_model[model] = {
            "cells": len(selected),
            "mirror_cost_min": min(row["mirror_cost"] for row in selected),
            "mirror_cost_max": max(row["mirror_cost"] for row in selected),
        }
    by_strategy = {}
    for strategy in strategies:
        selected = [row for row in rows if row["strategy"] == strategy]
        by_strategy[strategy] = {
            "cells": len(selected),
            "mirror_cost_min": min(row["mirror_cost"] for row in selected),
            "mirror_cost_max": max(row["mirror_cost"] for row in selected),
            "coverage_forward_min": min(row["coverage_forward"] for row in selected),
            "coverage_forward_max": max(row["coverage_forward"] for row in selected),
            "coverage_reversed_min": min(row["coverage_reversed"] for row in selected),
            "coverage_reversed_max": max(row["coverage_reversed"] for row in selected),
        }

    paired_model_gaps = []
    for strategy in strategies:
        for length in lengths:
            values = [row["mirror_cost"] for row in rows
                      if row["strategy"] == strategy
                      and row["n_letters"] == length]
            assert len(values) == len(models) == 2
            paired_model_gaps.append(abs(values[0] - values[1]))

    return {
        "status": "pass",
        "result_sha256": digest(path),
        "models": models,
        "strategies": strategies,
        "target_lengths": lengths,
        "spans_per_length": design["spans_per_length"],
        "cells": len(rows),
        "mirror_cost_min": min(costs),
        "mirror_cost_max": max(costs),
        "forward_resegmentation_cost_min": min(resegmentation),
        "forward_resegmentation_cost_max": max(resegmentation),
        "maximum_cross_model_cost_gap": max(paired_model_gaps),
        "by_model": by_model,
        "by_strategy": by_strategy,
        "checks": {
            "complete_factorial": True,
            "saved_spans_match_normalized_sources": True,
            "vocabulary_hash_and_size_match": True,
            "segmentations_preserve_letters": True,
            "coverage_recomputed": True,
            "reported_arithmetic_consistent": True,
            "model_logits_recomputed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", nargs="?", default=str(DEFAULT_RESULT))
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit(Path(args.result))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
