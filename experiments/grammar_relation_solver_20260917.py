"""Finite-state grammar relation search for readable letter palindromes.

This lane is deliberately not a completed-sentence generator followed by a
reverse.  It composes a typed slot grammar from both ends while maintaining a
small residual tape at the seam.  The residual is itself required to close as
a palindrome, so a seam may occur inside a word.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "grammar-relation-solver-20260917.json"
EXPERIMENT_ID = "grammar-relation-solver-20260917"
SIGNATURE = "typed-grammar-state|residual-character-debt|bidirectional-relation|authored-role-lattice"

BANKS = {
    "DET": ("a", "the", "this", "that", "one"),
    "NOUN": ("artist", "baker", "child", "dream", "friend", "garden", "letter", "river", "story", "teacher", "writer"),
    "VERB": ("admires", "answers", "carries", "delivers", "guards", "helps", "keeps", "learns", "reads", "writes"),
    "ADV": ("again", "often", "quietly", "still", "today", "well"),
    "PREP": ("about", "after", "in", "near", "with"),
}
PATTERNS = {
    "reported_scene": ("DET", "NOUN", "VERB", "DET", "NOUN", "ADV", "VERB", "DET", "NOUN"),
    "linked_scene": ("DET", "NOUN", "VERB", "PREP", "DET", "NOUN", "ADV", "VERB", "DET", "NOUN"),
}
FUNCTION = frozenset("a an the this that one again often still today well about after in near with".split())

def tape(s: str) -> str:
    return "".join(re.findall("[a-z]", s.lower()))

def independent_audit(s: str) -> dict:
    t = tape(s)
    mismatches = [(i, len(t)-1-i) for i in range(len(t)//2) if t[i] != t[-i-1]]
    f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "exact": bool(t) and not mismatches,
            "mismatches": mismatches[:8], "sha256_forward": f, "sha256_reverse": r}

def shortcuts(words: tuple[str, ...]) -> dict:
    content = [w for w in words if w not in FUNCTION]
    return {"repeated_content": len(content) != len(set(content)),
            "self_palindromic_words": [w for w in words if len(w) > 1 and w == w[::-1]],
            "word_order_mirror": list(words) == [w[::-1] for w in reversed(words)]}

@dataclass
class Result:
    candidates: list[dict]
    rejected: list[dict]
    states: int
    mismatches: int
    best: dict | None

def solve(slots: tuple[str, ...], *, budget=120_000) -> Result:
    # (left slot, right slot, active left word/offset, active right word/offset,
    # assignment, used content).  Offsets are character debt, not token debt.
    stack = [(0, len(slots)-1, None, 0, None, 0, (None,)*len(slots), frozenset())]
    seen = set(); candidates=[]; rejected=[]; mismatches=0; states=0; best=None
    def add(words, reason="closure"):
        nonlocal candidates, rejected
        rendered = " ".join(words)
        audit = independent_audit(rendered); sc = shortcuts(words)
        row = {"rendered": rendered, "words": words, "audit": audit,
               "shortcuts": sc, "provenance": {"experiment": EXPERIMENT_ID,
               "pattern": slots, "construction": "live bidirectional typed relation",
               "reason": reason}}
        if audit["exact"] and not any(sc.values()): candidates.append(row)
        elif audit["exact"]: rejected.append(row)
    while stack and states < budget:
        li, ri, lw, lp, rw, rp, assign, used = stack.pop(); states += 1
        if lw is not None and lp == len(lw):
            stack.append((li+1,ri,None,0,rw,rp,assign,used)); continue
        if rw is not None and rp == 0:
            stack.append((li,ri,lw,lp,None,0,assign,used)); continue
        key=(li,ri,lw,lp,rw,rp,assign,used)
        if key in seen: continue
        seen.add(key)
        realized=sum(x is not None for x in assign)
        if best is None or realized > best["slots_realized"]:
            best={"slots_realized":realized,"assignment":assign,"pattern":slots}
        if li > ri:
            if all(x is not None for x in assign): add(tuple(assign), "grammar relation closure")
            continue
        if li == ri and lw is None and rw is None:
            for word in BANKS[slots[li]]:
                if len(tape(word)) == 1:
                    a=list(assign); a[li]=word; add(tuple(a), "one-character center")
            continue
        if li == ri and lw is None and rw is not None:
            residual=tape(rw)[:rp]
            if residual and residual == residual[::-1]:
                add(tuple(x if x is not None else rw for x in assign), "palindromic right residual")
            continue
        if li == ri and rw is None and lw is not None:
            residual=tape(lw)[lp:]
            if residual and residual == residual[::-1]:
                add(tuple(x if x is not None else lw for x in assign), "palindromic left residual")
            continue
        # Start a typed word on either side. Content words are unique.
        if lw is None:
            for w in BANKS[slots[li]]:
                if w not in used or w in FUNCTION:
                    a=list(assign); a[li]=w
                    stack.append((li,ri,w,0,rw,rp,tuple(a),used | ({w} if w not in FUNCTION else set())))
            continue
        if rw is None:
            for w in BANKS[slots[ri]]:
                if w not in used or w in FUNCTION:
                    a=list(assign); a[ri]=w
                    stack.append((li,ri,lw,lp,w,len(tape(w)),tuple(a),used | ({w} if w not in FUNCTION else set())))
            continue
        lt=tape(lw); rt=tape(rw)
        if lp < len(lt) and rp > 0:
            if lt[lp] != rt[rp-1]:
                mismatches += 1; continue
            stack.append((li,ri,lw,lp+1,rw,rp-1,assign,used))
    return Result(candidates, rejected, states, mismatches, best)

def main():
    rows=[]
    for name, pattern in PATTERNS.items():
        r=solve(pattern)
        rows.append({"pattern":name,"slots":pattern,"states":r.states,"mismatch_edges":r.mismatches,
                     "budget_exhausted":r.states>=120_000,"best_partial":r.best,
                     "candidates":r.candidates,"rejected_exact":r.rejected,
                     "next_repair":"add seam-local typed morphology substitutions while retaining relation state"})
    OUT.write_text(json.dumps({"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,
        "status":"completed_relation_search","independent_validator":"independent_audit",
        "patterns":rows}, indent=2)+"\n")
    print(json.dumps({"experiment_id":EXPERIMENT_ID,"patterns":[(x["pattern"],x["states"],len(x["candidates"])) for x in rows]}, indent=2))
if __name__ == "__main__": main()
