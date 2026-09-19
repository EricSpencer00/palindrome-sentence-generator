"""Search exact mirrors by resegmenting independently attested phrase units.

Unlike whole-span reversal, each side is assembled from two intact corpus
phrase units.  The mirrored side may use a different boundary between units;
thus the character equation is solved before surface segmentation.  This is
an experiment and never a readability certificate.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks
from llm_palindrome.validator import is_palindrome, normalize


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def units(*, min_letters: int, max_letters: int, limit: int) -> list[dict]:
    data = json.loads((ROOT / "data/ngrams_wikitext2.json").read_text())
    rows = []
    seen = set()
    for key in ("3", "4", "5", "6"):
        for rank, phrase in enumerate(data[key]):
            text = " ".join(phrase.casefold().split())
            tokens = text.split()
            # Retain attested subphrases as units.  These are not invented
            # word lists: each unit is a contiguous slice of a frozen ngram,
            # with the parent ngram retained as provenance.
            for start in range(len(tokens)):
                for end in range(start + 1, min(len(tokens), start + 2) + 1):
                    subtext = " ".join(tokens[start:end])
                    tape = letters(subtext)
                    if not (min_letters <= len(tape) <= max_letters):
                        continue
                    if tape in seen:
                        continue
                    seen.add(tape)
                    rows.append({"text": subtext, "tape": tape,
                                 "words": tuple(subtext.split()),
                                 "source": "frozen WikiText-2 ngram subspan",
                                 "parent": text, "rank": rank, "ngram": key})
                    if len(rows) >= limit:
                        return rows
    return rows


def audit(text: str) -> dict:
    checks = mechanical_admission_checks(text)
    return {"exact": is_palindrome(text), "normalized": normalize(text),
            "admission": checks}


def run(min_letters: int, max_letters: int, limit: int, max_pairs: int) -> dict:
    pool = units(min_letters=min_letters, max_letters=max_letters, limit=limit)
    by_tape: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        by_tape[row["tape"]].append(row)
    candidates = []
    controls = []
    exact = 0
    admitted = 0
    checked = 0
    # Three units per side, with independently chosen boundaries.  The pair
    # index keeps this bounded while allowing 1/2-word units at every join.
    pair_index: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for a in pool:
        for b in pool:
            tape = a["tape"] + b["tape"]
            if len(tape) <= 38:
                pair_index[tape].append((a, b))
    for a in pool:
        for b in pool:
            for e in pool:
                tape = a["tape"] + b["tape"] + e["tape"]
                if len(tape) < 39:
                    continue
                rev = tape[::-1]
                for split in range(1, len(rev)):
                    lefts = by_tape.get(rev[:split])
                    rights = pair_index.get(rev[split:])
                    if not lefts or not rights:
                        continue
                    for c in lefts:
                        for d, f in rights:
                            checked += 1
                            text = f"{a['text']} {b['text']} {e['text']} {c['text']} {d['text']} {f['text']}"
                            result = audit(text)
                            if not result["exact"]:
                                raise AssertionError("character equation produced non-palindrome")
                            exact += 1
                            row = {"text": text, "length": len(result["normalized"]),
                                   "units": [a, b, e, c, d, f], "audit": result,
                                   "boundary_split": split}
                            candidates.append(row)
                            if result["admission"].get("eligible", False):
                                admitted += 1
                            if len(candidates) >= max_pairs:
                                return {"pool": len(pool), "checked": checked,
                                        "exact": exact, "admitted": admitted,
                                        "candidates": candidates}
    # Preserve an intact-prose control even when the exact intersection is
    # empty; it makes the failed lane auditable rather than a silent miss.
    if len(pool) >= 6:
        control_text = " ".join(row["text"] for row in pool[:6])
        controls.append({"text": control_text, "audit": audit(control_text),
                         "units": pool[:6]})
    return {"pool": len(pool), "checked": checked, "exact": exact,
            "admitted": admitted, "candidates": candidates,
            "controls": controls}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-letters", type=int, default=5)
    p.add_argument("--max-letters", type=int, default=16)
    p.add_argument("--limit", type=int, default=1600)
    p.add_argument("--max-pairs", type=int, default=100)
    args = p.parse_args()
    result = run(args.min_letters, args.max_letters, args.limit, args.max_pairs)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("pool", "checked", "exact", "admitted")}))


if __name__ == "__main__":
    main()
