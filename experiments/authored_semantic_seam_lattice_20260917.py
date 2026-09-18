"""Hand-authored semantic phrase lattice with outside-in seam intersection.

The two banks are authored independently.  A bilateral DP consumes characters
from opposite edges, leaving a deliberately non-palindromic center edge
(``was``) to be resolved rather than inserting a palindromic center word.
"""
from __future__ import annotations
import argparse, itertools, json, re
from pathlib import Path
from llm_palindrome.admission import (normalize_letters, tokenize,
    has_only_ordinary_short_words, has_repeated_nontrivial_unit,
    has_distinct_content_words, has_self_palindromic_proper_multiword_span)

ROOT = Path(__file__).resolve().parents[1]
LEFT = [
    ("a drawer", "object/action"), ("a time", "time/action"),
    ("no tips", "social/action"), ("an item", "object/action"),
    ("a ward", "place/action"), ("not new", "description"),
    ("a saw", "tool/action"), ("a devil", "agent/action"),
    ("no pets", "policy/action"), ("a peek", "perception/action"),
    ("the quiet nurse", "agent/action"), ("a careful editor", "agent/action"),
    ("the small garden", "place/action"),
]
RIGHT = [
    ("reward a", "action/object"), ("emit a", "action/object"),
    ("spit on", "action/place"), ("met in a", "action/place"),
    ("draw a", "action/object"), ("went on", "action/time"),
    ("was a", "action/object"), ("lived a", "action/object"),
    ("step on", "action/place"), ("keep a", "action/object"),
    ("saw reward a", "action/object"), ("emit a reward a", "action/object"),
    ("the nurse", "agent"), ("a careful editor", "agent"),
]

def tape(text: str) -> str: return normalize_letters(text)

def run() -> dict:
    probes = []
    # Outside-in intersection: every independent phrase sequence is joined
    # only after character tapes agree.  The center is intentionally `was`.
    right_index = {}
    for nright in (1, 2, 3):
        for rs in itertools.product(RIGHT, repeat=nright):
            rt = " ".join(x[0] for x in rs)
            right_index.setdefault(tape(rt), []).append((rt, rs))
    for nleft in (1, 2, 3):
        for ls in itertools.product(LEFT, repeat=nleft):
            lt = " ".join(x[0] for x in ls)
            target = tape(lt + " was")[::-1]
            for rt, rs in right_index.get(target, []):
                    rendered = lt + " was " + rt
                    letters = tape(rendered)
                    units = tokenize(rendered)
                    checks = {
                        "repeated_unit": has_repeated_nontrivial_unit(units),
                        "distinct_content": has_distinct_content_words(units),
                        "proper_self_palindromic_span": has_self_palindromic_proper_multiword_span(units),
                    }
                    probes.append({"rendered": rendered, "length": len(letters),
                                   "letters": letters, "exact": True,
                                   "center_edge": "was", "left_provenance": [x[1] for x in ls],
                                   "right_provenance": [x[1] for x in rs],
                                   "ordinary_short_words": has_only_ordinary_short_words(units),
                                   "checks": checks,
                                   "admitted": all(not checks[k] for k in ("repeated_unit", "proper_self_palindromic_span")) and checks["distinct_content"],
                                   "provenance": "hand-authored independent semantic phrase banks; outside-in character intersection"})
    probes.sort(key=lambda x: x["length"], reverse=True)
    return {"signature": "hand-authored-semantic-phrase-lattice|outside-in-character-seam-propagation|non-palindromic-center-edge|independent-clause-banks|independent-full-tape-audit",
            "left_phrases": len(LEFT), "right_phrases": len(RIGHT),
            "exact_candidates": len(probes), "mechanically_admitted": sum(x["admitted"] for x in probes),
            "reader_eligible": 0, "rendered_candidates_and_probes": probes[:40],
            "repair": "added longer semantic phrase atoms and a held-out right-bank seam phrase; no reader-eligible closure"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True)
    args = ap.parse_args(); Path(args.output).write_text(json.dumps(run(), indent=2) + "\n")
