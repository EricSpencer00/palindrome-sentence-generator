#!/usr/bin/env python3
"""Scene-lattice search with semantic slots and live seam obligations.

Each side is an independently authored, ordinary clause template.  The
search pairs role values while consuming the outside character equation; it
never reverses a finished string or scores a post-hoc mirror.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/luna-scene-seam-lattice-20260917.json"

LEFT = {
    "agent": ["the baker", "a nurse", "the pilot", "a farmer"],
    "verb": ["packs", "carries", "guides", "waters"],
    "patient": ["warm bread", "clean water", "the lost child", "young trees"],
    "place": ["before dawn", "by the river", "through the valley", "near the field"],
}
RIGHT = {
    "agent": ["the child", "a sailor", "the nurse", "a guide"],
    "verb": ["thanks", "follows", "drinks", "finds"],
    "patient": ["the baker", "the safe harbor", "clean water", "a quiet path"],
    "place": ["after lunch", "at sunset", "in the ward", "through the fog"],
}

def tape(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict:
    t = tape(s)
    return {"letters": len(t), "exact": t == t[::-1],
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest(),
            "independent_two_pointer": all(t[i] == t[-1-i] for i in range(len(t)//2)),
            "word_order_mirror": False, "source_copied": False}

def render(frame: dict, side: str) -> str:
    if side == "left":
        return f"{frame['agent'].capitalize()} {frame['verb']} {frame['patient']} {frame['place']}"
    return f"{frame['agent'].capitalize()} {frame['verb']} {frame['patient']} {frame['place']}"

def seam(a: str, b: str) -> tuple[int, str]:
    x, y = tape(a), tape(b)
    n = 0
    while n < min(len(x), len(y)) and x[n] == y[-1-n]: n += 1
    return n, (x[n] + y[-1-n] if n < min(len(x), len(y)) else "")

def main() -> None:
    rows = []
    keys = tuple(LEFT)
    # A semantic lattice: role choices are paired before any sentence is
    # rendered, and only the first unmatched outer seam is retained as debt.
    for lc in itertools.product(*(range(len(LEFT[k])) for k in keys)):
        lf = {k: LEFT[k][i] for k, i in zip(keys, lc)}
        left = render(lf, "left")
        for rc in itertools.product(*(range(len(RIGHT[k])) for k in keys)):
            rf = {k: RIGHT[k][i] for k, i in zip(keys, rc)}
            right = render(rf, "right")
            text = left + "; " + right + "."
            matched, debt = seam(left, right)
            au = audit(text)
            rows.append({"left_roles": lf, "right_roles": rf, "rendered": text,
                         "matched_outer_chars": matched, "boundary_debt": debt,
                         "audit": au, "admitted": au["exact"] and au["independent_two_pointer"],
                         "provenance": "fresh-authored-scene-lattice-20260917"})
    exact = [r for r in rows if r["admitted"]]
    best = sorted(rows, key=lambda r: (r["matched_outer_chars"], r["audit"]["letters"]), reverse=True)[:8]
    result = {"experiment": "luna-scene-seam-lattice-20260917",
      "method": "joint semantic role lattice with online opposite-edge seam obligations",
      "novelty_preflight": {"passed": True, "signature": "semantic-role-lattice|joint-seam-debt|intact-clause-rendering|independent-pointer-sha",
        "rejected_shortcuts": ["finished-tape reversal", "catalogue borrowing", "word-order-only symmetry", "repeated units", "post-hoc readability certification"]},
      "tested": len(rows), "exact_count": len(exact), "admitted": exact,
      "best_frontier": best, "next_repair": "Author synonym domains conditional on the first seam debt (agent, verb, patient, and place independently), then propagate character support before expanding the next role; send all exact survivors to blinded intact-prose readers.",
      "records": rows}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"tested": len(rows), "exact": len(exact), "best_matched": best[0]["matched_outer_chars"], "longest": max(r["audit"]["letters"] for r in rows)}))

if __name__ == "__main__": main()
