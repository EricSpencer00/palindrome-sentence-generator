"""Typed hand-authored clause automaton with live character obligations.

Each side is an independently lexicalized clause.  Expansion proceeds through
typed semantic slots; after every pair of slot choices, the two tapes are
compared from opposite ends and an exact residual character debt is retained.
This is a construction search, not a completed-text readability filter.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks

SLOTS = ("DET_SUBJ", "VERB", "DET_OBJ", "ADJUNCT")
DOMAINS = {
    "DET_SUBJ": (("the", "SG"), ("a", "SG"), ("our", "PL"), ("one", "SG"),
                 ("this", "SG"), ("my", "SG")),
    "VERB": (("carries", "SG", "TRANS"), ("draws", "SG", "TRANS"),
              ("helps", "SG", "TRANS"), ("marks", "SG", "TRANS"),
              ("reads", "SG", "TRANS"), ("sends", "SG", "TRANS"),
              ("carry", "PL", "TRANS"), ("draw", "PL", "TRANS")),
    "DET_OBJ": (("a", "SG"), ("one", "SG"), ("our", "PL"),
                ("the", "SG"), ("my", "SG")),
    "ADJUNCT": (("home", "INTRANS"), ("today", "INTRANS"), ("outside", "INTRANS"),
                ("at noon", "INTRANS"), ("by dawn", "INTRANS"), ("in town", "INTRANS")),
}
SUBJECTS = {"SG": ("baker", "doctor", "farmer", "guard", "teacher", "writer", "pilot"),
            "PL": ("bakers", "doctors", "farmers", "guards", "teachers", "writers", "pilots")}
OBJECTS = {"SG": ("letter", "message", "map", "memo", "parcel", "story", "signal"),
           "PL": ("letters", "messages", "maps", "memos", "parcels", "stories", "signals")}

def norm(s: str) -> str: return normalize_letters(s)
def sha(x) -> str: return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()

def choices(slot, side_state):
    """Yield typed lexical choices, enforcing agreement/valency at expansion."""
    subj_num = side_state.get("subj_num")
    if slot == "DET_SUBJ":
        for det, num in DOMAINS[slot]:
            for noun in SUBJECTS[num]: yield det + " " + noun, {"subj_num": num}
    elif slot == "VERB":
        for word, num, val in DOMAINS[slot]:
            if subj_num == num: yield word, {"verb_valency": val}
    elif slot == "DET_OBJ":
        for det, num in DOMAINS[slot]:
            for noun in OBJECTS[num]: yield det + " " + noun, {"obj_num": num}
    else:
        for word, val in DOMAINS[slot]: yield word, {"adjunct_valency": val}

def consume(debt_side, debt, left_piece, right_piece):
    """Compare newly appended left tape with newly appended reversed right tape."""
    a = debt + (norm(left_piece) if debt_side != "R" else norm(left_piece))
    b = norm(right_piece)[::-1]
    # debt is always the unmatched prefix from the prior comparison; append on
    # its owning side, then compare the two available prefixes.
    if debt_side == "R": a, b = norm(left_piece), debt + b
    k = min(len(a), len(b))
    if a[:k] != b[:k]: return None
    if len(a) > len(b): return "L", a[k:]
    if len(b) > len(a): return "R", b[k:]
    return "", ""

def search(limit=500_000):
    # State includes semantic features on both independently authored clauses.
    states = {("", "", "", "", "", ""): ([], [], {}, {})}
    counts = [1]; pruned = defaultdict(int)
    for slot in SLOTS:
        nxt = {}
        for (side, debt, lt, rt, lk, rk), (left, right, lf, rf) in states.items():
            for lp, lfeat in choices(slot, lf):
                for rp, rfeat in choices(slot, rf):
                    # Object/verb typing is enforced by the state, not later.
                    if slot == "VERB" and (lfeat.get("verb_valency") != "TRANS" or rfeat.get("verb_valency") != "TRANS"):
                        continue
                    got = consume(side, debt, lp, rp)
                    if got is None:
                        pruned["character_obligation_conflict"] += 1; continue
                    ns, nd = got
                    nlf = {**lf, **lfeat}; nrf = {**rf, **rfeat}
                    # Dominance key retains one witness per live debt and type.
                    key = (ns, nd, lt + lp, rt + rp, nlf.get("subj_num", ""), nrf.get("subj_num", ""))
                    if key not in nxt: nxt[key] = (left + [lp], right + [rp], nlf, nrf)
                    if len(nxt) >= limit: break
                if len(nxt) >= limit: break
            if len(nxt) >= limit: break
        states = nxt; counts.append(len(states))
        if not states: break
    candidates = []
    for (side, debt, _lt, _rt, _lk, _rk), (left, right, lf, rf) in states.items():
        if side or debt: continue
        text = " ".join(left + list(reversed(right)))
        audit = mechanical_admission_checks(text)
        candidates.append({"text": text, "letters": len(norm(text)), "audit": audit,
                           "left_clause": left, "right_clause": right,
                           "provenance": "luna_typed_clause_automaton_20260917"})
    return {"method": "typed_clause_automaton", "slots": SLOTS, "counts": counts,
            "pruned": dict(pruned), "terminals": candidates, "limit": limit,
            "domain_sizes": {k: len(v) for k,v in DOMAINS.items()}}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=500000)
    ap.add_argument("--out", type=Path, default=Path("runs/luna-typed-clause-automaton-20260917.json")); a=ap.parse_args()
    r=search(a.limit); r["independent_exact"]=[norm(c["text"]) == norm(c["text"])[::-1] for c in r["terminals"]]; r["run_sha256"]=sha(r)
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(r, indent=2)+"\n")
    print(json.dumps({"counts":r["counts"],"terminals":len(r["terminals"]),"pruned":r["pruned"]}, indent=2))
if __name__ == "__main__": main()
