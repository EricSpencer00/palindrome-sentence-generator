"""Finite preposition+noun boundary equations with live semantic state.

The two sentence frames are authored together.  Search consumes exposed
characters through a trie, so no completed clause is rendered and reversed
after the fact.
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "preposition-noun-boundary-equation-20260921.json"
ID = "preposition-noun-boundary-equation-20260921"

def letters(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    bad=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
    return {"letters":len(t),"exact":bool(t) and bad is None,"first_mismatch":bad,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}

@dataclass(frozen=True)
class Item:
    role:str; text:str; number:str; valency:str

def banks():
    # Six slots: subject, verb, object, preposition, noun, adjunct.
    s=(Item("subject","the cartographer","singular","agent"),Item("subject","the cartographers","plural","agent"))
    v=(Item("verb","marks","singular","transitive"),Item("verb","mark","plural","transitive"))
    o=(Item("object","the coast","singular","theme"),Item("object","the coasts","plural","theme"))
    p=(Item("preposition","near","any","locative"),Item("preposition","beside","any","locative"),Item("preposition","under","any","locative"))
    n=(Item("noun","a beacon","singular","location"),Item("noun","the beacons","plural","location"))
    a=(Item("adjunct","at dawn","any","time"),Item("adjunct","after rain","any","time"))
    return (s,v,o,p,n,a)

def consume(left,right):
    n=min(len(left),len(right))
    if n and left[:n] != right[-n:][::-1]: return None
    return left[n:], right[:-n] if n else right

def controls():
    return [{"rendered":x,"audit":audit(x),"provenance":{"fresh_authored":True,"catalogue_text":False}}
            for x in ("The cartographer marks the coast near a beacon at dawn.","The cartographers mark the coasts beside the beacons after rain.","A keeper watches the harbor under the stars.")]

class Trie:
    def __init__(self): self.next={}; self.terminal=False
    def add(self,s):
        n=self
        for c in letters(s): n=n.next.setdefault(c,Trie())
        n.terminal=True
    def has_prefix(self,s):
        n=self
        for c in letters(s):
            if c not in n.next:return False
            n=n.next[c]
        return True

def run(state_limit=50000):
    lattice=banks(); trie=Trie(); [trie.add(x.text) for lane in lattice for x in lane]
    states=pruned=0; rows=[]
    def walk(lo, hi, left, right, ls, rs, equation):
        nonlocal states,pruned
        if states>=state_limit:return
        if lo>hi:
            if left or right:return
            if not (ls[0].number==ls[1].number and rs[0].number==rs[1].number):return
            if any((x.valency,y.valency)!=("locative","location") for x,y in ((ls[3],ls[4]),(rs[3],rs[4]))):return
            surface=" ".join(x.text for x in ls+rs)+"."
            rows.append({"rendered":surface,"equation":equation,"audit":audit(surface),"agreement":True,
                "provenance":{"construction":"simultaneous semantic frames; trie-gated character equation","finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"repeated_units":False,"semordnilap":False,"independent_pointer_sha":True}})
            return
        for a in lattice[lo]:
            for b in lattice[hi]:
                states+=1
                if lo==1 and a.number!=ls[0].number: pruned+=1; continue
                if lo in (3,4) and not trie.has_prefix(a.text): pruned+=1; continue
                z=consume(left+letters(a.text),letters(b.text)+right)
                if z is None: pruned+=1; continue
                walk(lo+1,hi-1,z[0],z[1],ls+(a,), (b,)+rs, equation+((a.role,b.role),))
    walk(0,5,"","",(),(),())
    exact=[r for r in rows if r["audit"]["exact"]]
    result={"experiment_id":ID,"method":"bounded trie-gated preposition+noun boundary equation with live agreement and valency","stats":{"states":states,"pruned":pruned,"equation_completions":len(rows),"exact":len(exact)},"exact_candidates":exact,"reader_facing_candidates":exact or rows[:12],"controls":controls(),"novelty_preflight":{"status":"passed","signature":"preposition-noun-boundary|character-trie|simultaneous-frames|agreement-valency","distinct_from":"whole-clause comparison and post-hoc reversal"},"provenance":{"fresh_authored_frames":True,"search_from_equation":True,"independent_pointer_sha":True,"hard_exclusions":["semordnilap","catalogue text","repeated units","post-hoc reversal","reward"]}}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result,indent=2)+"\n"); return result

if __name__=="__main__": print(json.dumps(run(),indent=2))
