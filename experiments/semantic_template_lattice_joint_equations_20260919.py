"""Small human-authored semantic template lattice with joint edge equations.

This is a construction probe: clause roles are selected before realization, but
each realization is admitted only while its normalized letters satisfy the
outer-inward equation against the jointly selected mirror realization.
"""
from __future__ import annotations

import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCENES = [
    {"id": "scribe", "subject": "The scribe", "verb": "copies", "object": "the map", "adjunct": "at dawn"},
    {"id": "gardener", "subject": "A gardener", "verb": "waters", "object": "the roses", "adjunct": "by noon"},
    {"id": "pilot", "subject": "The pilot", "verb": "guides", "object": "a vessel", "adjunct": "through fog"},
]

def norm(s: str) -> str:
    return "".join(re.findall(r"[a-z]", s.lower()))

def live_equation(left: str, right: str) -> dict:
    """Compare left and right as opposite halves without reversing either tape."""
    a, b = norm(left), norm(right)
    n = min(len(a), len(b))
    matches = 0
    first = None
    for i in range(n):
        if a[i] == b[-1-i]:
            matches += 1
        elif first is None:
            first = {"left_offset": i, "right_offset": len(b)-1-i, "left": a[i], "right": b[-1-i]}
    return {"matched_edges": matches, "compared": n, "first_mismatch": first,
            "length_equal": len(a) == len(b), "equation_pass": first is None and len(a) == len(b)}

def audit(text: str) -> dict:
    t = norm(text)
    rev = t[::-1]
    return {"letters": len(t), "two_pointer_exact": all(x == y for x, y in zip(t, rev)),
            "sha256_normalized": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reversed": hashlib.sha256(rev.encode()).hexdigest(),
            "hash_exact": hashlib.sha256(t.encode()).hexdigest() == hashlib.sha256(rev.encode()).hexdigest()}

def main() -> None:
    rows = []
    for left in SCENES:
        for right in SCENES:
            left_text = f"{left['subject']} {left['verb']} {left['object']} {left['adjunct']}."
            right_text = f"{right['subject']} {right['verb']} {right['object']} {right['adjunct']}."
            prose = left_text + " " + right_text
            equation = live_equation(left_text, right_text)
            rows.append({"left_scene": left["id"], "right_scene": right["id"],
                         "roles": ["subject", "verb", "object", "adjunct"],
                         "rendered_prose": prose, "equation": equation,
                         "independent_audit": audit(prose),
                         "admitted": equation["equation_pass"] and audit(prose)["two_pointer_exact"]})
    payload = {"experiment": "semantic-template-lattice-joint-equations-20260919",
               "hypothesis": "joint semantic scene/valency selection can satisfy mirrored character equations",
               "method": {"scene_count": len(SCENES), "pair_count": len(rows), "composition": "subject+verb+object+adjunct complete clauses",
                          "anti_shortcut": ["no catalogue or seed sentences", "no reversal of rendered prose", "no fixed tape", "no generated prose filtering"],
                          "verification": "independent pointer and SHA-256 audit replayed from each rendered row"},
               "rows": rows,
               "stats": {"rows": len(rows), "equation_passes": sum(r["equation"]["equation_pass"] for r in rows),
                         "exact_palindromes": sum(r["admitted"] for r in rows),
                         "longest_letters": max(r["independent_audit"]["letters"] for r in rows)},
               "failure_frontier": "The first mismatch is exposed at the first mirrored edge for every pair; lexical choices need boundary-aware inflection and a residual character domain, not post-hoc clause pairing.",
               "next_repair": "Add a two-word subject/object lattice whose inflectional variants carry residual character domains across the clause boundary, while retaining semantic role typing."}
    out = ROOT / "runs" / "semantic-template-lattice-joint-equations-20260919.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], indent=2))

if __name__ == "__main__":
    main()
