"""Search independently authored prose across word and sentence boundaries.

This is deliberately not a phrase-bank reversal: left and right clause
sequences are selected independently, and the only live constraint is the
outer character obligation.  A score is a diagnostic; only a complete tape
passes the exact gate.
"""
from __future__ import annotations

import hashlib, itertools, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import is_palindrome

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "multispan-scene-boundary-search-20261001.json"

# Fresh, ordinary scene clauses.  No clause is paired with its reversal.
CLAUSES = [
    "At dawn, Mira opened the garden gate.",
    "A patient fox watched the quiet road.",
    "The baker carried warm bread to the school.",
    "Near the river, Lena found a blue button.",
    "The old map marked a path through the pines.",
    "By noon, the keeper had mended the lantern.",
    "A young teacher read the letter beside the fire.",
    "At dusk, the children brought water to the hens.",
]

def tape(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def audit(s: str) -> dict:
    t = tape(s)
    mm = next(((i, t[i], t[-1-i]) for i in range(len(t)//2)
               if t[i] != t[-1-i]), None)
    f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "two_pointer_exact": mm is None and bool(t),
            "validator_exact": is_palindrome(s), "first_mismatch": mm,
            "sha256_forward": f, "sha256_reverse": r, "sha_equal": f == r}

def matched_outer(a: str, b: str) -> int:
    x, y = tape(a), tape(b)
    return next((i for i in range(min(len(x), len(y))) if x[i] != y[-1-i]), min(len(x), len(y)))

def run() -> dict:
    rows = []
    rejected_repeats = []
    # Independent choices on each side; boundaries may fall inside a word
    # obligation because the comparison is character-level.
    for li in itertools.product(range(len(CLAUSES)), repeat=2):
        left = " ".join(CLAUSES[i] for i in li)
        for ri in itertools.product(range(len(CLAUSES)), repeat=2):
            if len(set(li + ri)) != 4:
                rejected_repeats.append({"left_indices": li, "right_indices": ri,
                                         "reason": "repeated clause unit"})
                continue
            right = " ".join(CLAUSES[i] for i in ri)
            score = matched_outer(left, right)
            lt, rt = tape(left), tape(right)
            rows.append({"left_indices": li, "right_indices": ri,
                         "left": left, "right": right,
                         "matched_outer_letters": score,
                         "crosses_word_boundary": score > 0 and score < len(lt) and score < len(rt),
                         "combined_audit": audit(left + " " + right)})
    rows.sort(key=lambda x: (x["combined_audit"]["two_pointer_exact"], x["matched_outer_letters"]), reverse=True)
    best = rows[0]
    exact = [r for r in rows if r["combined_audit"]["two_pointer_exact"]]
    return {"experiment": "multispan_scene_boundary_search_20261001",
            "method": "independent two-clause scene sequences with live character obligations and free word/sentence boundaries",
            "search": {"clause_count": len(CLAUSES), "left_sequences": len(CLAUSES)**2,
                       "right_sequences": len(CLAUSES)**2, "states": len(rows),
                       "rejected_repeated_units": len(rejected_repeats),
                       "exact_closures": len(exact)},
            "best_frontier": best,
            "rendered_candidates": rows[:5],
            "rejected_repeated_units": rejected_repeats[:32],
            "provenance": {"clauses": CLAUSES, "fresh_authored": True,
                           "catalogue_text": False, "finished_tape_reversal": False,
                           "repeated_self_palindromic_unit": False, "repeated_units": False,
                           "punctuation_changes_letters": False,
                           "reader_gate": "closed: no human ratings; frontier is diagnostic",
                           "novelty": "independent multiword spans cross sentence/word boundaries; no semordnilap terminal requirement"},
            "next_construction": {"operator": "author a third clause against the exact residual suffix at the deepest cross-word frontier, then re-run bilateral chart",
                                  "deepest_span": best["matched_outer_letters"],
                                  "reason": "two-clause scene sequences do not close; preserve the actual prose frontier rather than reversing a finished tape"}}

if __name__ == "__main__":
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    print(OUT)
