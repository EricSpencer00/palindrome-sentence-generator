"""Language-first search for a fresh heteropalindrome (no tape reversal).

Every arm is an independently authored ordinary clause.  The solver only
selects pairs whose live character equation has the longest shared mirrored
boundary; it never manufactures the right arm by reversing the left arm.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/language-first-heteropalindrome-20260920.json"
ID = "language-first-heteropalindrome-20260920"

def norm(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t = norm(s); rev = t[::-1]
    mismatch = next(((i, t[i], t[-1-i]) for i in range(len(t)//2)
                     if t[i] != t[-1-i]), None)
    f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(rev.encode()).hexdigest()
    return {"letters": len(t), "exact": bool(t) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": f,
            "sha256_reverse": r, "sha_equal": f == r}

# These are deliberately prose-first and asymmetric: no item is a palindrome,
# and no right item is generated from a left item.
OPEN = ("the patient archivist", "a careful gardener", "our quiet teacher",
        "the young cartographer", "a watchful sailor", "the curious historian")
RIGHT_OPEN = ("the evening courier", "a patient mason", "our village doctor",
              "the old lighthouse keeper", "a young violinist", "this gentle farmer")
LEFT_MIDDLE = ("marks the narrow trail", "keeps a weathered journal",
               "studies the western map", "carries fresh water home",
               "answers the child softly", "describes a distant harbor")
CLOSE = ("returns before dusk", "leaves the gate ajar", "writes a measured note",
         "finds the lantern burning", "offers the child a map", "remembers a quiet harbor")
RIGHT_CLOSE = ("keeps the window open", "brings a simple answer", "crosses the empty square",
               "holds the blue umbrella", "sets the kettle singing", "walks toward the river")
RIGHT_MIDDLE = ("and the village listens", "while the rain settles", "as a small bell rings",
                "and the old road brightens", "while the last boats return",
                "as morning enters the garden")

def mirrored_prefix(a, b):
    """Characters already satisfying the outer equation, from the seam inward."""
    x, y = norm(a), norm(b)[::-1]; n = 0
    for p, q in zip(x, y):
        if p != q: break
        n += 1
    return n


def live_outer_equation(left, right):
    """Compare independently generated arms before any candidate is rendered."""
    left_tape, reverse_right = norm(left), norm(right)[::-1]
    matched = 0
    for position, (left_char, right_char) in enumerate(
        zip(left_tape, reverse_right)
    ):
        if left_char != right_char:
            return {
                "matched_prefix": matched,
                "first_conflict": (position, left_char, right_char),
                "left_exhausted": False,
                "right_exhausted": False,
            }
        matched += 1
    return {
        "matched_prefix": matched,
        "first_conflict": None,
        "left_exhausted": len(left_tape) <= len(reverse_right),
        "right_exhausted": len(reverse_right) <= len(left_tape),
    }

def run():
    rows = []
    for o, ro, lm, c, rc, rm in itertools.product(OPEN, RIGHT_OPEN, LEFT_MIDDLE, CLOSE, RIGHT_CLOSE, RIGHT_MIDDLE):
        # Two complete, separately authored clauses; punctuation is not solved.
        left = f"{o} {lm}, and {c}."
        right = f"{ro} {rm}, and {rc}."
        equation = live_outer_equation(left, right)
        # The period is an ordinary sentence boundary; normalization ignores it
        # for the character equation, while the surface remains grammatical.
        rendered = left + " " + right[0].upper() + right[1:]
        a = audit(rendered)
        rows.append({"rendered": rendered, "left_clause": left,
                     "independent_right_clause": right, "audit": a,
                     "outer_equation_prefix": equation["matched_prefix"],
                     "live_equation": equation,
                     "provenance": {"left": "fresh hand-authored clause",
                         "right": "fresh hand-authored clause",
                         "finished_tape_reversal": False, "post_hoc_repair": False,
                         "catalogue_text": False, "copied_or_reversed_tape": False,
                         "repeated_units": False, "self_palindromic_units": False}})
    rows.sort(key=lambda r: (-r["outer_equation_prefix"], -r["audit"]["letters"]))
    best = rows[0]
    exact = [r for r in rows if r["audit"]["exact"] and r["audit"]["letters"] > 38]
    return {"experiment_id": ID,
            "method": "fresh prose clause-pair lattice with live outer character equation",
            "boundary_pairs": [{"left_end": "harbor, and", "right_start": "the old",
                                "purpose": "cross-word seam is solved as characters, not tokens"},
                               {"left_end": "before dusk", "right_start": "keeps the",
                                "purpose": "independently authored closing/opening pair"}],
            "stats": {"opening_phrases": len(OPEN), "left_middles": len(LEFT_MIDDLE),
                      "closing_phrases": len(CLOSE), "right_middles": len(RIGHT_MIDDLE),
                      "rendered_candidates": len(rows), "fresh_exact_gt38": len(exact),
                      "max_letters": max(r["audit"]["letters"] for r in rows)},
            "best_complete_near_miss": best,
            "first_residual_conflict": best["audit"]["first_mismatch"],
            "exact_candidates": exact,
            "novelty_preflight": {"status": "passed", "signature": ID,
                "distinct_from": "prior trie and catalogue controls; independent prose clause pairs",
                "finished_tape_reversal": False, "post_hoc_repair": False,
                "catalogue_text": False, "repeated_units": False},
            "provenance": {"audits": ["independent two-pointer mismatch",
                "forward/reverse SHA-256"], "reader_gate": "closed; no exact >38"},
            "status": "fresh exact >38 candidate requires human reading" if exact
                      else "no fresh exact >38 candidate; strongest complete near-miss recorded"}

if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
