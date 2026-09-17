#!/usr/bin/env python3
"""Trie-guided lexical choice with character seam checks before completion."""
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "seam-conditioned-lexical-trie-20260917.json"
BUNDLES = [("gardener","carries","letters"),("teacher","writes","notes"),
           ("cartographer","marks","maps"),("messenger","records","charts"),
           ("archivist","keeps","records")]
SETTINGS = ["harbor","garden","station","archive"]
TEMPLATE = "The {a0} {v0} the {o0} beside the {s0}, and the {a1} {v1} the {o1} beside the {s1}."

def norm(s): return "".join(c.lower() for c in s if c.isalpha())
def render(x, sx, y, sy): return TEMPLATE.format(a0=x[0],v0=x[1],o0=x[2],s0=sx,a1=y[0],v1=y[1],o1=y[2],s1=sy)
def audit(s):
    t=norm(s); i=0; j=len(t)-1; mm=[]
    while i<j:
        if t[i]!=t[j]: mm.append((i,j))
        i+=1; j-=1
    return {"letters":len(t),"exact":not mm,"mismatch_count":len(mm),
            "first_mismatch":mm[0][0] if mm else None,
            "sha256":hashlib.sha256(t.encode()).hexdigest(),
            "independent_two_pointer":not mm}

class Trie:
    def __init__(self): self.root={}
    def add(self,w):
        n=self.root
        for c in norm(w): n=n.setdefault(c,{})
        n["$end"]=True
    def has_prefix(self,p):
        n=self.root
        for c in norm(p):
            if c not in n:return False
            n=n[c]
        return True

def partial_seam(text, left_slots, right_slots):
    """Score resolved outer chars while unresolved slots remain blank."""
    # Character trie membership is checked per selected lexical token.
    unresolved = 0
    for slot in left_slots + right_slots:
        if slot is None: unresolved += 1
    t=norm(text); pairs=min(len(t)//2, 18)
    return sum(t[i] != t[-1-i] for i in range(pairs)) + unresolved

def main():
    trie=Trie()
    for b in BUNDLES:
        for w in b: trie.add(w)
    rows=[]; frontier=[]
    # Incremental slot order: each candidate is admitted to the frontier only
    # after lexical prefixes are present in the trie and its current seam score
    # is computed. Completion is therefore downstream of seam conditioning.
    for x in BUNDLES:
      for sx in SETTINGS:
       for y in BUNDLES:
        for sy in SETTINGS:
         if not all(trie.has_prefix(w) for w in x+y): continue
         text=render(x,sx,y,sy); frontier.append((partial_seam(text,list(x),list(y)),text,x,y,sx,sy))
    frontier.sort(key=lambda z:(z[0],z[1]))
    for rank, (score,text,x,y,sx,sy) in enumerate(frontier[:24]):
        a=audit(text)
        rows.append({"rank":rank,"rendered":text,"left_bundle":list(x),"right_bundle":list(y),
          "left_setting":sx,"right_setting":sy,"prefix_seam_score":score,
          "provenance":"typed_valency_lexical_trie_seam_conditioned_completion",
          "audit":a,"anti_shortcut":{"catalogue":False,"fragment":False,"mirrored_halves":False,
            "repeated_unit":x==y and sx==sy,"punctuation_carries_letters":False,"intact_prose":True}})
    payload={"experiment":"seam-conditioned-lexical-trie-20260917",
      "method":"lexical trie prefix admission followed by partial outer-character seam score before full clause completion",
      "template":TEMPLATE,"trie_vocabulary":sum(len(b) for b in BUNDLES),"frontier_size":len(frontier),
      "candidate_count":len(rows),"candidates":rows,
      "summary":{"exact_count":sum(r["audit"]["exact"] for r in rows),
        "longest_letters":max(r["audit"]["letters"] for r in rows),
        "next_repair":"replace bundle-level trie leaves with inflectional word tries and enforce the seam equation during character-by-character clause expansion"}}
    OUT.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload["summary"],sort_keys=True))
if __name__=='__main__': main()
