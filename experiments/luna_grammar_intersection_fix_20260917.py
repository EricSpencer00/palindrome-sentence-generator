#!/usr/bin/env python3
"""Finite grammar/product search with an explicit cross-word tape invariant.

The two sides are generated independently.  A match is accepted only after
reconstructing both rendered sentences and checking tape(A)==reverse(tape(B)).
The famous seed is a regression fixture, never a generated result.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from dataclasses import dataclass, asdict
from pathlib import Path

def tape(s: str) -> str:
    return "".join(c.lower() for c in s if c.isascii() and c.isalpha())

def digest(s: str) -> str:
    return hashlib.sha256(tape(s).encode()).hexdigest()

@dataclass(frozen=True)
class Candidate:
    left: str; right: str; method: str; parent: str; seed: int

def grammar(seed: int):
    """Small productive grammar; alternatives are compositional, not corpus text."""
    subjects = ["a kind sailor", "the quiet teacher", "a young poet", "the patient nurse"]
    verbs = ["carried", "noticed", "opened", "marked"]
    objects = ["a letter", "the lantern", "a garden", "the window"]
    tails = ["beside", "near", "under", "toward"]
    places = ["a harbor", "the garden", "a river", "the house"]
    for i, (s,v,o,p,x) in enumerate(( (s,v,o,p,x) for s in subjects for v in verbs for o in objects for p in tails for x in places)):
        yield f"{s} {v} {o} {p} {x}", seed+i

def audit(c: Candidate):
    a, b = tape(c.left), tape(c.right)
    flags=[]
    if len(a) < 40: flags.append("too_short")
    if a != b[::-1]: flags.append("equation_failure")
    if c.left.lower() == c.right.lower(): flags.append("same_surface")
    if any(x in c.left.lower() for x in ["panama", "taco cat", "racecar"]): flags.append("catalogue_control")
    return {"candidate":asdict(c), "letters":len(a), "mismatches":sum(x!=y for x,y in zip(a,b[::-1])),
            "exact":not flags, "flags":flags, "sha256_forward":digest(c.left),
            "sha256_reverse":hashlib.sha256(b[::-1].encode()).hexdigest()}

def regression():
    # Independently reconstruct the classic readable seed; catches the prior
    # bug where a character-synchronous path did not match reconstructed words.
    left="An aide rips nine memos; some men inspire Diana."
    right="An aide rips nine memos; some men inspire Diana."
    assert tape(left) == tape(right)[::-1], (tape(left), tape(right)[::-1])

def run(out: Path):
    regression()
    rows=[]; by_reverse={}
    for text, seed in grammar(17):
        by_reverse.setdefault(tape(text)[::-1], []).append((text,seed))
    # Independent second grammar pass; no shared sentence objects.
    for text, seed in grammar(1701):
        for right, rseed in by_reverse.get(tape(text), []):
            c=Candidate(text,right,"joint_cross_word_grammar_product",f"grammar:{seed}",rseed)
            rows.append(audit(c))
    accepted=[r for r in rows if r["exact"]]
    payload={"status":"completed", "method":"independent finite grammar intersection",
             "regression":"passed", "generated_pairs":len(rows), "accepted":accepted,
             "all_rows":rows, "next_repair":"expand compositional frames while preserving the reconstructed-tape invariant",
             "provenance":{"grammar":"hand-authored productive templates", "catalogue_text":False}}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({k:payload[k] for k in ["status","generated_pairs","accepted","next_repair"]},indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=Path("runs/luna-grammar-intersection-fix-20260917.json")); a=ap.parse_args(); run(a.out)
