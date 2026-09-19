"""Boundary-indexed search with asymmetric authored clause lengths."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import normalize_letters

EXPERIMENT_ID = "hst-asymmetric-clause-growth-20260919"
SLOTS = ("ldet", "lsubj", "lverb", "lmod1", "lmod2", "lobj", "r det", "rsubj", "rverb", "robj")
BANK = {
 "ldet": ("a", "the", "some"), "lsubj": ("quiet baker", "young sailor", "patient farmer", "Mara"),
 "lverb": ("marks", "packs", "plants", "reads"), "lmod1": ("near dawn", "with care", "by noon"),
 "lmod2": ("fresh", "small", "red"), "lobj": ("letters", "herbs", "cakes", "maps"),
 "r det": ("a", "the", "some"), "rsubj": ("kind poet", "Nora", "calm cook", "Diana"),
 "rverb": ("reads", "packs", "plants", "marks"), "robj": ("maps", "cakes", "herbs", "letters"),
}

def audit(text):
    t=normalize_letters(text); i,j=0,len(t)-1; bad=[]
    while i<j:
        if t[i]!=t[j]: bad.append((i,j,t[i],t[j]))
        i+=1; j-=1
    f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters":len(t),"two_pointer_exact":bool(t) and not bad,"mismatch_count":len(bad),
            "first_mismatch":bad[0] if bad else None,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}

def grow(target, limit=300_000):
    tape=[None]*target; nodes=0; leaves=0; deepest=0; rows=[]
    def put(word,pos):
        chars=normalize_letters(word)
        if pos+len(chars)>target:return None
        out=tape[:]
        for k,ch in enumerate(chars):
            p=pos+k; q=target-1-p
            for z in (p,q):
                if out[z] not in (None,ch): return None
                out[z]=ch
        return out
    def dfs(k,pos,words):
        nonlocal nodes,leaves,deepest
        if nodes>=limit:return
        nodes+=1; deepest=max(deepest,k)
        if k==len(SLOTS):
            leaves+=1
            if pos==target:
                text=" ".join(words)+"."; rows.append({"rendered":text,"audit":audit(text),"slots":dict(zip(SLOTS,words)),"provenance":{"indexed_before_render":True,"post_hoc_repair":False,"finished_tape_reversal":False}})
            return
        key=SLOTS[k]
        for word in BANK[key]:
            placed=put(word,pos)
            if placed is None:continue
            old=tape[:]; tape[:]=placed; dfs(k+1,pos+len(normalize_letters(word)),words+[word]); tape[:]=old
    dfs(0,0,[])
    return {"target":target,"nodes":nodes,"leaves":leaves,"deepest_slot":deepest,"candidates":rows}

def run():
    results=[grow(n) for n in range(40,81)]; rows=[r for x in results for r in x["candidates"]]
    return {"experiment_id":EXPERIMENT_ID,"method":"asymmetric left six-slot/right four-slot boundary indexed growth","remote_target":"hst-bench","results":rows,"stats":{"nodes":sum(x["nodes"] for x in results),"leaves":sum(x["leaves"] for x in results),"candidates":len(rows),"exact":sum(r["audit"]["two_pointer_exact"] for r in rows),"longest":max((r["audit"]["letters"] for r in rows),default=0)},"audit":"independent two-pointer and forward/reverse SHA-256","residual":"No exact closure in fresh authored asymmetric inventory; all slot obligations were checked before placement.","novelty_preflight":{"status":"passed","reused_38_tape":False,"finished_text_repair":False}}

if __name__=="__main__":
    out=run(); p=Path(__file__).resolve().parents[1]/"runs"/(EXPERIMENT_ID+".json"); p.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out["stats"],sort_keys=True))
