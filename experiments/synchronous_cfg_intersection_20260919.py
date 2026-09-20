"""Synchronous whole-grammar intersection (no post-hoc repair).

Two clauses are derived from the same typed grammar, but their words are chosen
at the same time.  Every newly exposed character is compared with its opposite
endpoint before recursion continues.  The output is therefore a derivation,
not a completed tape subsequently reversed or repaired.
"""
from __future__ import annotations
import hashlib, json, re
from collections import Counter
from pathlib import Path

def letters(s: str) -> str: return re.sub(r"[^a-z]", "", s.casefold())
def audit(s: str) -> dict[str, object]:
    t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters":len(t),"exact":bool(t) and t==t[::-1],"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}

def search(banks: dict[str, tuple[str,...]], limit: int=16) -> dict[str,object]:
    # A complete English grammar: determiner noun verb determiner noun prep noun.
    roles=("det","subject","verb","obj_det","object","prep","complement")
    states=pruned=0; candidates=[]
    def walk(lo,hi,prefix,suffix,left,right):
        nonlocal states,pruned
        if len(candidates)>=limit:return
        if lo>hi:
            states+=1; text=" ".join(left+right); a=audit(text)
            if a["exact"]: candidates.append({"rendered":text,"audit":a,"provenance":{"grammar":"DET NOUN VERB DET NOUN ADP NOUN","roles":roles,"construction":"synchronous character intersection"}})
            return
        for l in banks[roles[lo]]:
            if l in left or l in right or l==l[::-1]:continue
            for r in banks[roles[hi]]:
                if r in left or r in right or r==r[::-1] or r==l:continue
                lp,rs=prefix+letters(l),letters(r)+suffix
                states+=1
                n=min(len(lp),len(rs))
                if lp[:n]!=rs[::-1][:n]: pruned+=1; continue
                walk(lo+1,hi-1,lp,rs,left+(l,),(r,)+right)
    walk(0,len(roles)-1,"","",tuple(),tuple())
    return {"method":"synchronous-cfg-intersection-20260919","grammar":roles,"banks":{"sizes":{k:len(v) for k,v in banks.items()}},"stats":{"states":states,"pruned":pruned,"exact":len(candidates)},"candidates":candidates,"provenance":{"source":"Brown universal tagged corpus, frequency-ranked lexical partitions","no_repair":True,"no_finished_tape_reversal":True,"word_order_mirror":False}}

def main():
    from nltk.corpus import brown
    c={k:Counter() for k in ("det","subject","verb","obj_det","object","prep","complement")}
    for sent in brown.tagged_sents(tagset="universal")[:50000]:
        for i in range(len(sent)-6):
            tags=[x[1] for x in sent[i:i+7]]
            if tags[0]=="DET" and tags[1]=="NOUN" and tags[2] in {"VERB","AUX"} and tags[3]=="DET" and tags[4]=="NOUN" and tags[5]=="ADP" and tags[6]=="NOUN":
                vals=[x[0].casefold() for x in sent[i:i+7]]
                for k,v in zip(c,vals):
                    if v.isalpha():c[k][v]+=1
    banks={k:tuple(w for w,n in c[k].most_common(64)) for k in c}
    out=search(banks); Path("runs").mkdir(exist_ok=True); Path("runs/synchronous-cfg-intersection-20260919.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
if __name__=="__main__":main()
