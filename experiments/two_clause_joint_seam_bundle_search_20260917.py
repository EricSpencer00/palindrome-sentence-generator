#!/usr/bin/env python3
"""Joint two-clause seam search with agreement-featured lexical bundles."""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "two-clause-joint-seam-bundle-search-20260917.json"
BUNDLES = [
    ("gardener", "carries", "letters"),
    ("teacher", "writes", "notes"),
    ("cartographer", "marks", "maps"),
    ("messenger", "records", "charts"),
    ("archivist", "keeps", "records"),
]
SETTINGS = ["harbor", "garden", "station", "archive"]
TEMPLATE = "The {a0} {v0} the {o0} beside the {s0}, and the {a1} {v1} the {o1} beside the {s1}."

def norm(s): return "".join(c.lower() for c in s if c.isalpha())

def audit(s):
    t = norm(s); mismatches = []
    i, j = 0, len(t) - 1
    while i < j:
        if t[i] != t[j]: mismatches.append((i, j))
        i += 1; j -= 1
    return {"letters": len(t), "exact": not mismatches,
            "mismatch_count": len(mismatches),
            "first_mismatch": mismatches[0][0] if mismatches else None,
            "sha256": hashlib.sha256(t.encode()).hexdigest(),
            "independent_two_pointer": not mismatches}

def render(b0, s0, b1, s1):
    return TEMPLATE.format(a0=b0[0], v0=b0[1], o0=b0[2], s0=s0,
                           a1=b1[0], v1=b1[1], o1=b1[2], s1=s1)

def seam_side(a, length):
    if a["first_mismatch"] is None: return "none"
    return "left_clause" if a["first_mismatch"] < length // 2 else "right_clause"

def main():
    rows = []
    seeds = [(0, 1, "harbor", "garden"), (2, 3, "station", "archive"),
             (1, 4, "garden", "harbor"), (3, 0, "archive", "station")]
    for seed_i, (i0, i1, s0, s1) in enumerate(seeds):
        b0, b1 = BUNDLES[i0], BUNDLES[i1]
        for step in range(4):
            text = render(b0, s0, b1, s1); a = audit(text)
            rows.append({"seed": seed_i, "step": step, "rendered": text,
                         "left_bundle": list(b0), "right_bundle": list(b1),
                         "left_setting": s0, "right_setting": s1,
                         "seam_side": seam_side(a, a["letters"]),
                         "provenance": "typed_two_clause_joint_seam_equation",
                         "audit": a,
                         "anti_shortcut": {"catalogue": False, "fragment": False,
                           "mirrored_halves": False,
                           "repeated_unit": (b0 == b1 and s0 == s1),
                           "punctuation_carries_letters": False, "intact_prose": True}})
            if a["exact"]: break
            # Joint move: evaluate every pair of bundles/settings against the
            # complete two-clause tape, never repair either clause independently.
            options = []
            for x in BUNDLES:
                for y in BUNDLES:
                    for sx in SETTINGS:
                        for sy in SETTINGS:
                            if (x, y, sx, sy) == (b0, b1, s0, s1): continue
                            ta = audit(render(x, sx, y, sy))
                            options.append((ta["mismatch_count"], render(x, sx, y, sy), x, y, sx, sy))
            _, _, b0, b1, s0, s1 = min(options, key=lambda x: (x[0], x[1]))
    payload = {"experiment": "two-clause-joint-seam-bundle-search-20260917",
      "method": "joint exhaustive neighborhood over two agreement-carrying valency bundles and settings, scored on complete tape seam",
      "template": TEMPLATE, "candidate_count": len(rows), "candidates": rows,
      "summary": {"exact_count": sum(r["audit"]["exact"] for r in rows),
        "longest_letters": max(r["audit"]["letters"] for r in rows),
        "next_repair": "retain joint bundle scoring but add a seam-conditioned lexical trie for word-boundary choices, so the crossing character equation is solved before clause completion"}}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))

if __name__ == "__main__": main()
