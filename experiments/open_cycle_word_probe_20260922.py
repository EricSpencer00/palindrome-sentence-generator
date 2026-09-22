"""Mine common-word open morphology cycles for productive residuals."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from experiments.open_carrier_brown_probe_20260922 import ORDINARY_TWO, corpus_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("brown_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-residual", type=int, default=7)
    args = parser.parse_args()
    counts = Counter()
    proper = set()
    tags = defaultdict(Counter)
    for _source, _line, chunk in corpus_chunks(args.brown_root):
        for word, tag in chunk:
            counts[word] += 1
            tags[word][tag] += 1
            if tag.startswith("np") or tag.startswith("fw"):
                proper.add(word)
    words = {
        word for word, count in counts.items()
        if count >= args.min_count and word not in proper
        and (len(word) > 2 or word in ORDINARY_TWO)
        and word != word[::-1]
    }
    grouped = defaultdict(list)
    for y in words:
        exposed = y[::-1]
        for width in range(1, min(args.max_residual, len(y) - 1) + 1):
            # Residual strings are drawn from the bounded common-word alphabet;
            # testing prefixes of x would be circular, so use all short suffixes
            # of ``exposed`` and replay the stated equation exactly.
            residual = exposed[-width:]
            right = residual + exposed
            if not right.endswith(residual) or len(right) <= width:
                continue
            x = right[:-width]
            if x not in words or x == y:
                continue
            grouped[residual].append({
                "x": x, "y": y, "x_count": counts[x], "y_count": counts[y],
                "x_tags": dict(tags[x]), "y_tags": dict(tags[y]),
                "equation": {"left": x + residual, "right": right,
                             "holds": x + residual == right},
            })
    productive = []
    for residual, pairs in grouped.items():
        unique = {(row["x"], row["y"]): row for row in pairs}
        rows = sorted(unique.values(), key=lambda row: (
            -(row["x_count"] + row["y_count"]), row["x"], row["y"]))
        disjoint = []
        used = set()
        for row in rows:
            if row["x"] in used or row["y"] in used:
                continue
            disjoint.append(row)
            used.update((row["x"], row["y"]))
        if len(disjoint) >= 2:
            productive.append({"residual": residual, "pairs": disjoint[:20],
                               "disjoint_pair_count": len(disjoint)})
    productive.sort(key=lambda row: (
        -row["disjoint_pair_count"], len(row["residual"]), row["residual"]))
    payload = {
        "probe": "open-cycle-word-probe-20260922",
        "stats": {"word_types": len(words), "productive_residuals": len(productive),
                  "max_disjoint_pairs": max((r["disjoint_pair_count"] for r in productive), default=0)},
        "productive_residuals": productive,
        "provenance": {"source": "Brown lowercase word counts/tags",
                       "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "proper_tagged_words_excluded": True},
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    for row in productive[:30]:
        print(row["residual"], row["disjoint_pair_count"],
              [(p["x"], p["y"], p["x_count"] + p["y_count"]) for p in row["pairs"][:8]])


if __name__ == "__main__":
    main()
