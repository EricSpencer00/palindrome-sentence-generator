"""Fresh nested composition of the existing novel mirror-pair inventory.

Each selected unit contributes an authored ordinary-order left phrase and its
exact character-reverse right phrase.  Units are nested as L1 ... Lk C Rk ...
R1; the constructor never reverses a finished sentence.  Bigram score is only
a proposal rank, never an admission test.
"""
from __future__ import annotations
import hashlib, itertools, json, math, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = json.loads((ROOT / "data/novel_pairs.json").read_text())
KNOWN = set(json.loads((ROOT / "data/known_palindromes.json").read_text()))
WORD_RE = re.compile(r"[a-z]+")

def norm(s): return "".join(WORD_RE.findall(s.lower()))
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def audit(s):
    t=norm(s); i,j=0,len(t)-1
    while i<j and t[i]==t[j]: i+=1; j-=1
    return {"exact": i>=j, "first_mismatch": None if i>=j else [i,j,t[i],t[j]],
            "sha_forward":sha(t), "sha_reverse":sha(t[::-1]),
            "two_pointer_exact": i>=j, "letters":len(t)}

def words(seq): return [w for p in seq for w in p]
def score(text):
    # Proposal-only readability proxy: reward common short function words and
    # penalize adjacent repeated/awkward boundaries. It cannot admit a result.
    ws=WORD_RE.findall(text.lower()); common={"a","an","the","on","in","to","of","and","no","not","set","one","it"}
    return sum(2.0 if w in common else math.log1p(len(w)) for w in ws)-sum(2 for a,b in zip(ws,ws[1:]) if a==b)

def main():
    rows=[]; seen=set(); centers=("a","i","one")
    # k=3 and 4 are enough to exceed 38 letters while keeping the search broad.
    for k in (3,4):
      for ids in itertools.permutations(range(len(PAIRS)), k):
        ps=[PAIRS[i] for i in ids]
        lw=words([p["left"] for p in ps]); rw=words([p["right"] for p in ps[::-1]])
        if len(set(lw+rw)) != len(lw+rw): continue
        for c in centers:
          text=" ".join(lw+[c]+rw)
          a=audit(text)
          if a["letters"] < 39 or not a["exact"]: continue
          if norm(text) in KNOWN or text in seen: continue
          seen.add(text); rows.append({"text":text,"pair_ids":list(ids),"center":c,"proposal_score":score(text),"audit":a,
            "provenance":{"inventory":"data/novel_pairs.json","catalogue_used":False,"finished_tape_reversal":False,"word_order_mirror":False,"repeated_clause":False}})
    rows.sort(key=lambda x:(-x["proposal_score"],-x["audit"]["letters"]))
    out={"experiment":"nested-pair-chain-constructor-20260919","method":"L1...Lk C Rk...R1","candidates":rows[:100],"counts":{"exact":len(rows),"searched_k":[3,4]},"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/"runs").mkdir(exist_ok=True); (ROOT/"runs/nested-pair-chain-constructor-20260919.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"count":len(rows),"top":rows[:5]},indent=2))
if __name__ == "__main__": main()
