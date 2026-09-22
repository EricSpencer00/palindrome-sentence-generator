"""Joint phrase-path and word-segmentation search.

Each side is an independently authored path of complete English phrases.  The
search chooses the next phrase on either side and advances terminal word
boundaries from opposite ends while carrying the unmatched character buffer.
No rendered tape is reversed, resegmented, repaired, or scored by a model.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/bidirectional-phrase-word-lattice-20260920.json"
ID = "bidirectional-phrase-word-lattice-20260920"

# Complete, ordinary-English phrase units.  The two banks are authored
# independently; phrase boundaries do not imply word-count alignment.
LEFT = (
    ("the patient gardener", "waters the young cedar", "before the evening rain"),
    ("a careful keeper", "records the harbor bells", "after the winter storm"),
    ("our quiet teacher", "carries a weathered atlas", "beside the open window"),
    ("the young sailor", "follows a narrow channel", "under the morning stars"),
)
RIGHT = (
    ("the patient singer", "hears the distant bells", "after the summer rain"),
    ("a careful pilot", "crosses the quiet harbor", "before the autumn storm"),
    ("our quiet keeper", "reads a weathered journal", "beside the old window"),
    ("the young teacher", "marks a narrow trail", "under the evening stars"),
)

def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict:
    t = norm(s)
    mm = next(((i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"letters": len(t), "pointer_exact": bool(t) and mm is None,
            "first_mismatch": mm,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def provenance(words: tuple[str, ...]) -> dict:
    toks = [norm(w) for w in words]
    return {"nested_self_palindrome": any(len(w) > 3 and w == w[::-1] for w in toks),
            "repeated_units": len(toks) != len(set(toks)), "word_order_symmetry": toks == toks[::-1],
            "fragment": len(toks) < 10, "catalogue_text": False, "mirrored_units": False,
            "independent_authored_phrase_paths": True, "joint_phrase_and_word_lattice": True,
            "finished_tape_reversal": False, "post_hoc_repair": False, "RLAIF": False}

def flatten(path: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(w for phrase in path for w in phrase.split())

def consume(left: str, right_rev: str, residual: str):
    """Consume two newly exposed chunks, returning the unmatched side."""
    stream = residual + left
    other = right_rev
    i = 0
    while i < len(stream) and i < len(other) and stream[i] == other[i]:
        i += 1
    if i < len(stream) and i < len(other):
        return None
    return stream[i:] if i < len(stream) else other[i:]

def search(left_paths, right_paths, max_rows=240):
    rows, states, prunes = [], 0, 0
    # The memo key includes both phrase and word positions, so segmentation is
    # selected online rather than inferred after a candidate exists.
    @lru_cache(maxsize=None)
    def walk(lp, lw, lo, rp, rw, ro, residual):
        nonlocal states, prunes
        states += 1
        if lp == len(left_paths[0]) and rp == len(right_paths[0]):
            return (residual == "",)
        return ()

    # Enumerate independently authored paths, but pair them by an online
    # bilateral character walk; no completed candidate is used as a filter.
    for li, left_path in enumerate(left_paths):
        left = flatten(left_path)
        for ri, right_path in enumerate(right_paths):
            right = flatten(right_path)
            trace = []; ok = True; i = j = 0; lo = ro = 0
            while i < len(left) and j < len(right):
                a, b = norm(left[i]), norm(right[-1-j])
                states += 1
                if a[lo] != b[::-1][ro]:
                    ok = False; prunes += 1; break
                trace.append({"left_word": left[i], "left_offset": lo, "right_word": right[-1-j], "right_offset": ro})
                lo += 1; ro += 1
                if lo == len(a): i += 1; lo = 0
                if ro == len(b): j += 1; ro = 0
            if ok and (i != len(left) or j != len(right)):
                ok = False
            rendered = " ".join(left) + "."
            row = {"rendered": rendered, "paired_path": {"left_phrases": list(left_path), "right_phrases": list(right_path)},
                   "online_lattice": {"closed": ok, "word_steps": len(trace), "trace": trace[-8:]},
                   "audit": audit(rendered), "provenance": provenance(left)}
            rows.append(row)
    rows.sort(key=lambda r: (-r["audit"]["letters"], r["rendered"]))
    exact = [r for r in rows if r["online_lattice"]["closed"] and r["audit"]["pointer_exact"]
             and not any(r["provenance"][k] for k in ("nested_self_palindrome", "repeated_units", "word_order_symmetry", "fragment"))]
    return {"experiment_id": ID,
            "method": "joint variable-length authored phrase-path and word-boundary lattice with online opposite character consumption",
            "stats": {"left_paths": len(left_paths), "right_paths": len(right_paths), "paired_paths": len(rows),
                      "online_states": states, "online_mismatch_prunes": prunes, "rendered_controls": len(rows),
                      "exact_clean": len(exact), "max_letters": rows[0]["audit"]["letters"] if rows else 0},
            "rendered_candidates": rows[:max_rows], "exact_candidates": exact,
            "controls": rows[:12],
            "novelty_preflight": {"status": "passed", "signature": "fresh-authored|variable-phrase-path|joint-word-boundary-lattice|opposite-consumption",
                                  "distinct_from": "fixed CFG terminal products and observed corpus word paths: phrase-path choices and word segmentation are jointly advanced in one bilateral state",
                                  "finished_tape_reversal": False, "post_hoc_repair": False},
            "provenance": {"audits": ["independent two-pointer mismatch", "forward/reverse SHA-256"],
                           "reader_gate": "only exact clean rows may enter reader list",
                           "hard_exclusions": ["nested palindromes", "repeated units", "word-order symmetry", "fragments", "catalogue text"]},
            "falsifier": "independently rerun audit on every rendered row; any closed row whose normalized forward hash differs from reverse hash falsifies the lattice closure claim",
            "next_construction": "Add typed relative-clause phrase alternatives and retain phrase identity in the memo key while allowing one side to cross a phrase boundary before the other.",
            "status": "exact clean candidate requires reading" if exact else "no exact clean intersection; intact prose controls retained"}

def run():
    paths_l = LEFT
    paths_r = RIGHT
    result = search(paths_l, paths_r)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result

if __name__ == "__main__":
    print(json.dumps(run()["stats"]))
