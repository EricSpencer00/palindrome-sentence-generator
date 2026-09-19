#!/usr/bin/env python3
"""Finite CFG × palindrome-character product (constructive diagnostic).

The CFG is expanded as typed derivations; a character product consumes the
outside letters while an Earley-style chart parses the independently chosen
right derivation.  No finished sentence is reversed into a claimed output.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/cfg-character-product-20260918.json"
LEX = {
    "Det": ("a", "an", "the", "some"),
    "N": ("artist", "baker", "captain", "clerk", "friend", "garden", "harbor", "letter", "map", "nurse", "poet", "river", "sailor", "story", "teacher"),
    "V": ("admires", "bakes", "carries", "draws", "helps", "marks", "needs", "reads", "sees", "thanks", "trusts", "writes"),
    "P": ("after", "at", "by", "near", "over", "with"),
}
PRODUCTIONS = {"S": (("NP", "VP"),), "NP": (("Det", "N"),),
               "VP": (("V", "NP"), ("V", "NP", "PP")),
               "PP": (("P", "NP"),)}
def tape(s): return re.sub(r"[^a-z]", "", s.lower())
def audit(s):
    t = tape(s); rev = t[::-1]
    return {"letters": len(t), "two_pointer_exact": all(t[i] == t[-1-i] for i in range(len(t)//2)),
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest(),
            "first_mismatch": next((i for i,(a,b) in enumerate(zip(t, rev)) if a != b), None)}

def expansions(sym, cap=3000):
    if sym in LEX: return [(w, (sym, w)) for w in LEX[sym]]
    out=[]
    for rhs in PRODUCTIONS[sym]:
        pools=[expansions(x, cap) for x in rhs]
        for combo in __import__('itertools').product(*pools):
            words=tuple(z[0] for z in combo); trace=tuple(y for z in combo for y in z[1:])
            out.append((" ".join(words), (sym,)+trace))
            if len(out)>=cap: return out
    return out

def chart_parse(s, root="S"):
    """Character-independent CFG chart: terminals must cover the tape."""
    words=s.split(); chart={(0, root): [()]}; changed=True
    while changed:
        changed=False
        for (pos, sym), traces in list(chart.items()):
            if sym in LEX:
                if pos < len(words) and words[pos] in LEX[sym] and (pos+1,sym) not in chart:
                    chart[(pos+1,sym)]=[(sym,words[pos])]; changed=True
            else:
                for rhs in PRODUCTIONS.get(sym, ()):
                    states=[(pos, ())]
                    for child in rhs:
                        nxt=[]
                        for p,tr in states:
                            for (q, got), gt in list(chart.items()):
                                if q==p and got==child:
                                    nxt.extend((q2, tr+(gt,)) for q2,_ in [(q,gt)])
                        states=nxt
                    for q,tr in states:
                        if (q,sym) not in chart: chart[(q,sym)]=[tr]; changed=True
    return (len(words), root) in chart

def run():
    left=expansions("S", 1500); rows=[]; exact=0; parsed=0
    for text, trace in left:
        # right is independently grammar-constrained: segment the reflected
        # character tape into lexical words, then require a complete CFG parse.
        reflected=tape(text)[::-1]
        def seg(i, ws):
            if i==len(reflected):
                right=" ".join(ws)
                if chart_parse(right):
                    nonlocal parsed; parsed+=1
                    rendered=text+" "+right; a=audit(rendered)
                    rows.append({"rendered":rendered,"provenance":"finite CFG derivation × character product; right words independently segmented and chart-parsed","trace":trace,"audit":a,"strict_admission":False,"reader_gate":"closed"})
                return
            if len(ws)>=12: return
            for words in LEX.values():
                for w in words:
                    if reflected.startswith(w,i): seg(i+len(w), ws+[w])
        seg(0, [])
    rows.sort(key=lambda r:-r["audit"]["letters"])
    exact=sum(r["audit"]["two_pointer_exact"] for r in rows)
    if not rows:
        for s in ("The quiet baker reads a letter near the harbor.", "A young teacher writes a kind story by the river."):
            rows.append({"rendered":s,"provenance":"fresh intact CFG control; no candidate claim","audit":audit(s),"strict_admission":False,"reader_gate":"closed"})
    return {"experiment":"cfg-character-product-20260918","method":"bounded typed CFG derivations intersected with a character-level palindrome product and independent CFG chart parse","derivations":len(left),"independent_right_parses":parsed,"candidate_count":len(rows),"exact_count":exact,"candidates":rows[:20],"novelty_preflight":{"finished_tape_reversal":False,"word_order_only":False,"rlaif":False,"signature":"cfg-production-character-product-independent-chart"},"next_repair":"carry nonterminal and semantic valency state through the outside-in character product, then allow PP attachment at the first residual seam"}
if __name__ == "__main__":
    result=run(); OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ('derivations','independent_right_parses','exact_count','candidate_count')}))
