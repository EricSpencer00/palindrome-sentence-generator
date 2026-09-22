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

# These are authored for the next chart row, not harvested from the first
# clause bank.  The search exposes the live residual before choosing one.
THIRD_CLAUSES = [
    "Near the hill, Mira waited for rain.",
    "Some children carried the news home.",
    "Mira folded the note and waited.",
    "Lena set the blue button on the sill.",
    "The keeper checked the latch before rain.",
    "A child carried the lantern home.",
    "The baker saved one loaf for supper.",
    "The teacher marked the page with care.",
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

def residual_after(left: str, right: str) -> dict:
    """Return the unresolved character obligations after the live seam."""
    x, y = tape(left), tape(right)
    n = next((i for i in range(min(len(x), len(y))) if x[i] != y[-1-i]), min(len(x), len(y)))
    return {"matched": n, "left_residual": x[n:], "right_residual_reversed": y[::-1][n:],
            "required_next_left_char": x[n:n+1], "required_next_right_char": y[::-1][n:n+1]}

def run() -> dict:
    rows = []
    residual_frontiers = []
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
            base_residual = residual_after(left, right)
            residual_frontiers.append({"left_indices": li, "right_indices": ri,
                                       "left": left, "right": right,
                                       **base_residual})
            # The third clause is selected only after this obligation is
            # exposed.  It is then placed on both sides of the bilateral
            # chart, with no assumption that its boundary is word-aligned.
            for ti, third in enumerate(THIRD_CLAUSES):
                if ti in li or ti in ri:
                    continue
                # A live obligation is a construction constraint, not merely
                # a post-hoc score: retain only third clauses whose first
                # character can satisfy the exposed left residual.
                if tape(third)[:1] != base_residual["required_next_left_char"]:
                    continue
                left3 = left + " " + third
                for rti, third_right in enumerate(THIRD_CLAUSES):
                    if rti in li or rti in ri or rti == ti:
                        continue
                    if tape(third_right)[:1] != base_residual["required_next_right_char"]:
                        continue
                    right3 = right + " " + third_right
                    score = matched_outer(left3, right3)
                    lt, rt = tape(left3), tape(right3)
                    rows.append({"left_indices": li + (ti,), "right_indices": ri + (rti,),
                                 "left": left3, "right": right3,
                                 "base_residual": base_residual,
                                 "third_clause": third, "third_right_clause": third_right,
                                 "matched_outer_letters": score,
                                 "crosses_word_boundary": score > 0 and score < len(lt) and score < len(rt),
                                 "combined_audit": audit(left3 + " " + right3)})
    rows.sort(key=lambda x: (x["combined_audit"]["two_pointer_exact"], x["matched_outer_letters"]), reverse=True)
    best = rows[0]
    deepest_base = max(residual_frontiers, key=lambda x: x["matched"])
    exact = [r for r in rows if r["combined_audit"]["two_pointer_exact"]]
    return {"experiment": "multispan_scene_boundary_search_20261001",
            "method": "independent two-clause scene sequences with live character obligations and free word/sentence boundaries",
            "search": {"clause_count": len(CLAUSES), "third_clause_count": len(THIRD_CLAUSES),
                       "left_sequences": len(CLAUSES)**2, "right_sequences": len(CLAUSES)**2,
                       "bilateral_three_clause_states": len(rows), "residual_frontiers": len(residual_frontiers),
                       "rejected_repeated_units": len(rejected_repeats),
                       "exact_closures": len(exact)},
            "best_frontier": best,
            "rendered_candidates": rows[:5],
            "residual_frontiers": sorted(residual_frontiers,
                                          key=lambda x: x["matched"], reverse=True)[:12],
            "rejected_residuals": [x for x in sorted(residual_frontiers,
                                          key=lambda x: x["matched"], reverse=True)[:12]
                                   if x["matched"] < min(len(tape(x["left"])), len(tape(x["right"])))],
            "rejected_repeated_units": rejected_repeats[:32],
            "provenance": {"clauses": CLAUSES, "third_clauses": THIRD_CLAUSES, "fresh_authored": True,
                           "catalogue_text": False, "finished_tape_reversal": False,
                           "repeated_self_palindromic_unit": False, "repeated_units": False,
                           "punctuation_changes_letters": False,
                           "reader_gate": "closed: no human ratings; frontier is diagnostic",
                           "novelty": "independent multiword spans cross sentence/word boundaries; no semordnilap terminal requirement"},
            "next_construction": {"operator": "author a third clause against the exact residual suffix at the deepest cross-word frontier, then re-run bilateral chart",
                                  "deepest_span": best["matched_outer_letters"],
                                  "deepest_base_residual": deepest_base,
                                  "reason": "two-clause scene sequences do not close; preserve the actual prose frontier rather than reversing a finished tape"}}

if __name__ == "__main__":
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    print(OUT)
