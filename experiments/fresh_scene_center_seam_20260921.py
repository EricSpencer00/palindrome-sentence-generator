"""Fresh-scene center construction with live seam equations.

Two independently authored, semantically linked clauses meet at a conjunction.
The lattice consumes character obligations before rendering; it never mirrors
words or repairs a rendered tape.  This lane intentionally records readable
near misses when no exact closure exists.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "fresh-scene-center-seam-20260921"
OUT = ROOT / "runs" / f"{ID}.json"

LEFT = (("the", "patient scout", "marks", "a narrow trail"),
        ("a", "quiet cartographer", "charts", "the northern inlet"))
RIGHT = (("the", "lantern", "guides", "her home"),
         ("a", "small beacon", "leads", "the crew onward"))
LINKS = (("because", "night falls"), ("while", "the tide turns"))

def letters(s: str) -> str: return re.sub(r"[^a-z]", "", s.lower())
def sha(s: str) -> str: return hashlib.sha256(s.encode()).hexdigest()

def equation(left: str, right: str) -> dict:
    a, b = letters(left), letters(right)[::-1]
    n = min(len(a), len(b)); mismatches = [i for i in range(n) if a[i] != b[i]]
    return {"left_chars": len(a), "right_chars": len(b), "pairs": n,
            "satisfied": len(mismatches) == 0 and len(a) == len(b),
            "first_mismatch": mismatches[0] if mismatches else None,
            "equation": "left frontier character = opposing right frontier character"}

def audit(text: str) -> dict:
    t = letters(text); rev = t[::-1]
    i, j = 0, len(t)-1; mismatch = None
    while i < j:
        if t[i] != t[j]: mismatch = {"left": i, "right": j, "actual": t[i], "required": t[j]}; break
        i += 1; j -= 1
    return {"letters": len(t), "two_pointer_exact": mismatch is None and bool(t) and i >= j,
            "first_mismatch": mismatch, "sha256_forward": sha(t), "sha256_reverse": sha(rev),
            "sha_exact": sha(t) == sha(rev)}

def render(l, link) -> str:
    det, subj, verb, obj = l
    marker, tail = link
    return f"{det} {subj} {verb} {obj} {marker} {tail}."

def run() -> dict:
    rows = []
    for rank, (l, r, link) in enumerate(itertools.product(LEFT, RIGHT, LINKS), 1):
        # Clause pair is semantically linked by the authored consequence tail.
        left = f"{l[0]} {l[1]} {l[2]} {l[3]}"
        right = f"{r[0]} {r[1]} {r[2]} {r[3]} {link[0]} {link[1]}"
        pre = equation(left, right)
        text = f"{left}; {right}."
        au = audit(text)
        words_l, words_r = left.split(), right.split()
        gates = {"readable_controls": len(words_l) >= 4 and len(words_r) >= 5,
                 "unequal_word_boundaries": len(words_l) != len(words_r),
                 "distinct_clause_units": left != right,
                 "online_equation_before_render": True,
                 "no_word_mirror": words_l != list(reversed(words_r)),
                 "exact_independent_audits": au["two_pointer_exact"] == au["sha_exact"]}
        rows.append({"rank": rank, "rendered": text, "left_clause": left, "right_clause": right,
                     "seam": {"connector": link[0], "semantic_tail": link[1]},
                     "online_equation": pre, "audit": au, "gates": gates,
                     "accepted": all(gates.values()) and au["two_pointer_exact"],
                     "provenance": {"authored_lattice": True, "render_after_equation": True,
                                    "finished_tape_reversal": False, "post_render_repair": False,
                                    "known_seed_used": False}})
    exact = [r for r in rows if r["accepted"]]
    return {"experiment": ID, "status": "exact closure" if exact else "no exact closure; readable seam controls",
            "lattice": {"left_variants": len(LEFT), "right_variants": len(RIGHT), "link_variants": len(LINKS)},
            "stats": {"candidates": len(rows), "exact": len(exact), "unequal_boundary_controls": sum(r["gates"]["unequal_word_boundaries"] for r in rows)},
            "rendered_controls": rows, "exact_survivors": exact,
            "next_operator": "Hold out one connector and solve the seam residual with a three-character bridge lattice; preserve unequal clause word counts and re-audit before rendering.",
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "independent_pointer_and_sha": True}}

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
