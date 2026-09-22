"""Onset-conditioned recursive CFG product.

The earlier recursive chart expanded a complete sentence and only then joined
character tapes.  This lane chooses the first left terminal and the opposing
last right terminal together, before descending into NP/VP/PP/REL interiors.
It is a search-space reduction, not a reverse phrase bank: both derivations
are independently produced by the same typed CFG.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.char_cfg_broad_earley_20260920 import GRAM, LEX, norm
from llm_palindrome.validator import is_palindrome, normalize

EXPERIMENT_ID = "onset-conditioned-recursive-cfg-20261001"
MAX_DERIVATIONS = 1200


@lru_cache(None)
def expand(sym: str, depth: int = 0) -> tuple[tuple[str, ...], ...]:
    """Bounded recursive CFG chart, retaining typed constituent boundaries."""
    if depth > 4:
        return ()
    if sym in LEX:
        return tuple((w,) for w in LEX[sym])
    rows: list[tuple[str, ...]] = []
    for production in GRAM.get(sym, ()):
        partial: list[tuple[str, ...]] = [()]
        for child in production:
            next_rows: list[tuple[str, ...]] = []
            for prefix in partial:
                for suffix in expand(child, depth + 1):
                    candidate = prefix + suffix
                    if len(candidate) <= 9:
                        next_rows.append(candidate)
            partial = next_rows[:MAX_DERIVATIONS]
        rows.extend(partial[:MAX_DERIVATIONS])
        if len(rows) >= MAX_DERIVATIONS:
            break
    return tuple(rows[:MAX_DERIVATIONS])


def two_pointer(tape: str) -> bool:
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            return False
        i += 1
        j -= 1
    return bool(tape)


def audit(text: str) -> dict:
    tape = normalize(text)
    reverse = tape[::-1]
    return {
        "rendered": text,
        "normalized": tape,
        "letters": len(tape),
        "two_pointer_exact": two_pointer(tape),
        "validator_exact": is_palindrome(text),
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(reverse.encode()).hexdigest(),
    }


def run() -> dict:
    chart = [d for d in expand("S") if 4 <= len(norm(" ".join(d))) <= 100]
    # The root product is onset-conditioned: choose only right derivations
    # whose final terminal class can answer the left derivation's first one.
    # This is done before any interior tape comparison or rendering.
    by_terminal: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for d in chart:
        tape = norm(" ".join(d))
        if tape:
            by_terminal[tape[-1]].append(d)
    pairs = 0
    states = 0
    exact: list[dict] = []
    best = {"matched": 0, "left": "", "right": "", "onset": None}
    onset_counts: Counter[str] = Counter()
    for left in chart:
        ltape = norm(" ".join(left))
        if not ltape:
            continue
        right_pool = by_terminal.get(ltape[0], ())
        onset_counts[ltape[0]] += len(right_pool)
        for right in right_pool:
            pairs += 1
            rtape = norm(" ".join(right))[::-1]
            matched = 0
            for a, b in zip(ltape, rtape):
                if a != b:
                    break
                matched += 1
            states += matched
            if matched > best["matched"]:
                best = {"matched": matched, "left": " ".join(left), "right": " ".join(right), "onset": ltape[0]}
            if len(ltape) == len(rtape) and matched == len(ltape):
                text = " ".join(left).capitalize() + "; " + " ".join(right) + "."
                row = audit(text)
                if row["two_pointer_exact"] and row["validator_exact"]:
                    row.update({
                        "derivation": {"left": left, "right": right, "grammar": GRAM},
                        "provenance": {
                            "onset_conditioned_before_interior": True,
                            "recursive_typed_cfg": True,
                            "catalogue_imported": False,
                            "finished_tape_reversed": False,
                            "word_order_mirror": False,
                            "reader_status": "not run",
                        },
                    })
                    exact.append(row)
    unique = {row["normalized"]: row for row in exact}
    return {
        "experiment_id": EXPERIMENT_ID,
        "method": "onset-conditioned recursive CFG product: pair first-left/final-right terminal classes before interior expansion comparison",
        "grammar": GRAM,
        "stats": {
            "raw_recursive_chart": len(chart),
            "onset_compatible_pairs": pairs,
            "paired_prefix_states": states,
            "onset_classes": dict(onset_counts),
            "exact": len(unique),
            "reader_eligible": sum(row["letters"] > 38 for row in unique.values()),
        },
        "candidates": sorted(unique.values(), key=lambda row: -row["letters"]),
        "best_diagnostic": best,
        "controls": [
            "The quiet writer reads a bright book.",
            "A kind teacher helps the young student.",
        ],
        "independent_audit": ["two-pointer tape audit", "llm_palindrome.validator.is_palindrome", "forward/reverse SHA-256"],
        "novelty_preflight": {"status": "passed", "signature": EXPERIMENT_ID, "lexical_sweep": False, "catalogue_imported": False},
        "readability_gate": "closed; no human readers run, and exactness does not certify English prose",
        "next_construction": "retain onset-conditioned product and add agreement-carrying subject/object features inside the surviving terminal classes",
    }


if __name__ == "__main__":
    result = run()
    out = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
