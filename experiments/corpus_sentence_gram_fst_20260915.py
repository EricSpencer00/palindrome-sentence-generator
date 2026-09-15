"""Rank-partitioned sentence-gram phrase lattice palindrome experiment.

This is deliberately a phrase-token FST experiment, rather than a larger word
beam: left material is selected from one corpus rank partition and the
reverse residual is segmented by a disjoint held-out partition.  Exactness is
audited from the rendered string, independently of the lattice.
"""
from __future__ import annotations

import json, re
from functools import lru_cache
from pathlib import Path
from llm_palindrome.admission import normalize_letters, tokenize, has_only_ordinary_short_words

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "data/ngrams_wikitext2.json").read_text())

def phrases(max_words: int, partition: int, repair: int) -> list[tuple[str, str, int]]:
    rows: list[str] = []
    # sentence ngrams are the material source; lower ngrams are only the
    # explicit repair, never silently mixed into the base run.
    keys = ["sent10", "sent8", "sent6"] if repair == 0 else (["sent8", "sent6", "5", "4"] if repair == 1 else ["sent6", "5", "4", "3"])
    for key in keys:
        rows.extend(DATA.get(key, []))
    out = []
    seen = set()
    for rank, text in enumerate(rows):
        words = tuple(tokenize(text))
        if not words or len(words) > max_words or len(words) < 2:
            continue
        # Partitioning is by source rank, so no source phrase is available to
        # both sides.  This prevents mirrored phrase reuse.
        if rank % 2 != partition:
            continue
        tape = normalize_letters("".join(words))
        if len(tape) < 8 or tape in seen:
            continue
        seen.add(tape)
        out.append((" ".join(words), tape, rank))
    return out

def solve(repair: int) -> dict:
    left = phrases(10, 0, repair)
    right = phrases(5 if repair < 2 else 4, 1, repair)
    by_initial: dict[str, list[tuple[str, str, int]]] = {}
    for row in right:
        by_initial.setdefault(row[1][0], []).append(row)
    rows = []
    partials = []
    # A single-letter center keeps the construction honest: all prose is
    # supplied independently by the two phrase lattices.
    for text, tape, rank in left[:2500]:
        target = tape[::-1]
        @lru_cache(None)
        def segment(pos: int):
            if pos == len(target): return ()
            for phrase, ptape, prank in by_initial.get(target[pos], ()):
                if target.startswith(ptape, pos):
                    tail = segment(pos + len(ptape))
                    if tail is not None:
                        return ((phrase, prank),) + tail
            return None
        seg = segment(0)
        if seg is None:
            pos = 0; used = []
            while pos < len(target):
                choices = [x for x in by_initial.get(target[pos], ()) if target.startswith(x[1], pos)]
                if not choices: break
                choice = max(choices, key=lambda x: len(x[1]))
                used.append(choice); pos += len(choice[1])
            if used:
                attempted = text + " a " + " ".join(x[0] for x in used)
                partials.append({"rendered": attempted, "letters": normalize_letters(attempted),
                                 "exact": False, "length": len(tape) * 2 + 1,
                                 "left_rank": rank, "right_ranks": [x[2] for x in used],
                                 "matched_reverse_letters": pos,
                                 "target_reverse_letters": len(target),
                                 "provenance": "held-out phrase-lattice partial; repair required"})
            elif len(partials) < 40:
                attempted = text + " a"
                partials.append({"rendered": attempted, "letters": normalize_letters(attempted),
                                 "exact": False, "length": len(tape) * 2 + 1,
                                 "left_rank": rank, "right_ranks": [],
                                 "matched_reverse_letters": 0,
                                 "target_reverse_letters": len(target),
                                 "provenance": "held-out phrase-lattice dead-end; repair required"})
            continue
        rendered = text + " a " + " ".join(x[0] for x in seg)
        letters = normalize_letters(rendered)
        exact = letters == letters[::-1]
        units = tokenize(rendered)
        rows.append({"rendered": rendered, "letters": letters, "exact": exact,
                     "length": len(letters), "left_rank": rank,
                     "right_ranks": [x[1] for x in seg],
                     "ordinary_short_words": has_only_ordinary_short_words(units),
                     "provenance": "Wikitext-2 sentence/n-gram rank partitions; center=a"})
        if len(rows) >= 40: break
    if not rows:
        rows = sorted(partials, key=lambda x: x["matched_reverse_letters"], reverse=True)[:40]
    return {"method": "rank-partitioned-corpus-sentence-gram phrase lattice",
            "repair": repair, "left_count": len(left), "right_count": len(right),
            "rendered_probes": rows, "exact_probes": sum(x["exact"] for x in rows),
            "mechanically_admitted": 0, "reader_eligible": 0}

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--repair", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    Path(args.output).write_text(json.dumps(solve(args.repair), indent=2) + "\n")

if __name__ == "__main__": main()
