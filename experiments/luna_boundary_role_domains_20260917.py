"""Boundary-conditioned lexical domains for the paired clause chart.

The grammar remains an ordinary transitive clause.  Words are chosen from
role-specific inventories; their boundary characters are propagated through a
live debt before inner roles are expanded.  This is a construction search,
not a completed-tape reversal or a readability-afterthought filter.
"""
from __future__ import annotations

import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks

ROLES = ("DET", "SUBJ", "VERB", "OBJ", "PREP", "PLACE")
# The inventories are deliberately ordinary words, with role and valency
# labels retained in provenance.  Boundary buckets are computed, not guessed.
LEXICON = {
    "DET": (("a", "DET"), ("an", "DET"), ("the", "DET"), ("our", "DET"), ("one", "DET"), ("this", "DET")),
    "SUBJ": (("artist", "N"), ("baker", "N"), ("carer", "N"), ("doctor", "N"), ("farmer", "N"), ("guard", "N"), ("pilot", "N"), ("teacher", "N"), ("writer", "N"), ("reader", "N"), ("poet", "N"), ("editor", "N")),
    "VERB": (("asks", "TRANS"), ("draws", "TRANS"), ("helps", "TRANS"), ("marks", "TRANS"), ("reads", "TRANS"), ("sends", "TRANS"), ("shows", "TRANS"), ("writes", "TRANS")),
    "OBJ": (("book", "N"), ("letter", "N"), ("map", "N"), ("memo", "N"), ("note", "N"), ("parcel", "N"), ("story", "N"), ("tale", "N"), ("report", "N"), ("plan", "N")),
    "PREP": (("at", "P"), ("by", "P"), ("in", "P"), ("near", "P"), ("on", "P"), ("over", "P")),
    "PLACE": (("home", "N"), ("office", "N"), ("school", "N"), ("town", "N"), ("garden", "N"), ("market", "N"), ("river", "N"), ("station", "N"), ("area", "N"), ("villa", "N"), ("mesa", "N"), ("plaza", "N")),
}

def letters(w: str) -> str:
    return normalize_letters(w)

def buckets(role: str):
    out = defaultdict(list)
    for word, tag in LEXICON[role]:
        s = letters(word)
        out[(s[0], s[-1])].append((word, tag))
    return {f"{a}:{b}": v for (a, b), v in sorted(out.items())}

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def extend(debt: str, side: str, token: str):
    """Consume known opposite-edge debt, returning side/debt or None."""
    t = letters(token)
    if side == "R":
        t = t[::-1]
    if not debt:
        return ("L", t) if side == "L" else ("R", t)
    # debt is always stored in the orientation of the next left-side tape.
    if side == "L":
        x, y = debt + t, ""
    else:
        x, y = "", debt + t
    # A word may only be compared once both sides have material.
    return side, (debt + t if side == "L" else debt + t)

def consume(left: str, right: str):
    n = min(len(left), len(right))
    if left[:n] != right[:n]:
        return None
    if len(left) > n: return "L", left[n:]
    if len(right) > n: return "R", right[n:]
    return "", ""

def search(limit=300000, min_letters=45):
    # Expand outer role pairs, preserving ordinary clause order in each side.
    # State tapes are only boundary obligations; complete text is rendered later.
    states = {("", "", (), (), "TRANS")}
    counts = [len(states)]
    pruned = defaultdict(int)
    for i, role in enumerate(ROLES):
        mirror = ROLES[-1-i]
        nxt = {}
        for side, debt, left_words, right_words, valency in states:
            for lw, ltag in LEXICON[role]:
                if role == "VERB" and ltag != valency: continue
                for rw, rtag in LEXICON[mirror]:
                    # Independent lexicalization: no token identity constraint.
                    if role == "VERB" and rtag != valency: continue
                    a, b = letters(lw), letters(rw)[::-1]
                    if side == "L": result = consume(debt + a, b)
                    elif side == "R": result = consume(a, debt + b)
                    else: result = consume(a, b)
                    if result is None:
                        pruned["boundary_character_conflict"] += 1; continue
                    ns, nd = result
                    key = (ns, nd, left_words + (lw,), right_words + (rw,), valency)
                    # Keep all witnesses up to the explicit bound; the lexical
                    # words, not a post-hoc score, define the chart state.
                    nxt.setdefault((ns, nd, left_words + (lw,), right_words + (rw,), valency), key)
                    if len(nxt) >= limit: break
                if len(nxt) >= limit: break
            if len(nxt) >= limit: break
        states = set(nxt.values()); counts.append(len(states))
        if not states: break
    candidates = []
    for side, debt, lw, rw, valency in states:
        if side or debt: continue
        # Render two semantically ordered clauses. This run does not present
        # the right clause as a generated palindrome; it is an exact candidate
        # only if its ordinary rendering independently verifies.
        text = " ".join(lw + rw)
        n = len(letters(text))
        if n < min_letters: continue
        audit = mechanical_admission_checks(text)
        candidates.append({"text": text, "letters": n, "audit": audit,
                           "independent_tape": letters(text) == letters(text)[::-1],
                           "provenance": {"method": "boundary_role_domains", "left_roles": ROLES,
                                          "right_roles": tuple(reversed(ROLES)), "left_words": lw, "right_words": rw}})
    return {"chart_counts": counts, "candidates": candidates,
            "pruned": dict(pruned), "limit": limit, "min_letters": min_letters,
            "role_boundary_buckets": {r: buckets(r) for r in ROLES},
            "grammar": ROLES}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=300000); ap.add_argument("--out", type=Path, default=Path("runs/luna-boundary-role-domains-20260917.json")); args = ap.parse_args()
    result = search(args.limit)
    result["run_sha256"] = digest(result)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"chart_counts": result["chart_counts"], "candidates": len(result["candidates"]), "pruned": result["pruned"]}, indent=2))
if __name__ == "__main__": main()
