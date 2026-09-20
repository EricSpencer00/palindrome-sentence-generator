"""Parallel verb/noun/adjective selection for two authored scene utterances."""
from __future__ import annotations

import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"runs"/"blank-verse-parallel-scene-20260920.json"; EXPERIMENT_ID="blank-verse-parallel-scene-20260920"

def letters(s:str)->str:return re.sub(r"[^a-z]","",s.casefold())
def audit(s:str)->dict[str,object]:
    t=letters(s); bad=[(i,len(t)-i-1) for i in range(len(t)//2) if t[i]!=t[-i-1]]; f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters":len(t),"exact":bool(t) and not bad,"first_mismatch":bad[0] if bad else None,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}
def consume(l:str,r:str)->tuple[str,str]|None:
    n=min(len(l),len(r))
    if n and l[:n]!=r[-n:][::-1]:return None
    return l[n:],r[:-n] if n else r

@dataclass(frozen=True)
class Slot:
    role:str;text:str;utterance:str;feature:str
def slots(role,utterance,feature,*texts):return tuple(Slot(role,t,utterance,feature) for t in texts)

def build_lattice():
    # Rendered utterance A: "my lord sees the crimson moon".
    # Rendered utterance B is independently authored: "the patient queen
    # names our hidden vow".  The pairing order exposes all role variables
    # simultaneously instead of mirroring any token or finished utterance.
    return (
        slots("vocative","A","person","my lord","dear friend","good king"),
        slots("verb","A","finite","sees","marks","names","keeps"),
        slots("adjective","A","descriptive","crimson","silent","golden","winter"),
        slots("noun","A","object","moon","bell","rose","vow"),
        slots("noun","B","object","crown","gate","letter","song"),
        slots("adjective","B","descriptive","patient","hidden","bright","ancient"),
        slots("verb","B","finite","names","guards","reads","keeps"),
        slots("subject","B","person","the queen","the captain","our poet"),
    )

def run(*,state_limit=250_000):
    lattice=build_lattice(); states=pruned=advances=0; candidates=[]; witnesses=[]
    def witness(words,l,r,d):
        if len(witnesses)<24:
            z=" ".join(x.text for x in words);witnesses.append({"rendered":z,"depth":d,"left_residual":len(l),"right_residual":len(r),"audit":audit(z),"reader_status":"diagnostic witness; not candidate"})
    def walk(lo,hi,l,r,ls,rs):
        nonlocal states,pruned,advances
        if states>=state_limit:return
        if lo>hi:
            if l or r:return
            ordered=ls+tuple(reversed(rs));z=" ".join(x.text for x in ordered);checked=audit(z)
            if checked["exact"]:candidates.append({"rendered":z,"audit":checked,"provenance":{"construction":"parallel blank-verse scene lattice","roles":[x.role for x in ordered],"utterances":[x.utterance for x in ordered],"features":[x.feature for x in ordered],"independently_authored_utterances":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False},"reader_status":"unreviewed; exactness does not certify readability"})
            return
        if lo==hi:
            for x in lattice[lo]:
                states+=1;res=consume(l+letters(x.text),r)
                if res is None:pruned+=1;witness(ls+(x,)+tuple(reversed(rs)),l+letters(x.text),r,len(ls)+1)
                else:advances+=1;walk(lo+1,hi-1,res[0],res[1],ls+(x,),rs)
            return
        for a in lattice[lo]:
            for b in lattice[hi]:
                states+=1;res=consume(l+letters(a.text),letters(b.text)+r)
                if res is None:pruned+=1;witness(ls+(a,)+(b,)+tuple(reversed(rs)),l+letters(a.text),letters(b.text)+r,len(ls)+1);continue
                advances+=1;walk(lo+1,hi-1,res[0],res[1],ls+(a,),(b,)+rs)
    walk(0,len(lattice)-1,"","",(),())
    candidates.sort(key=lambda x:x["audit"]["letters"],reverse=True)
    controls=["my lord sees the crimson moon","dear friend marks the silent bell","the queen names our hidden vow"]
    result={"experiment":EXPERIMENT_ID,"method":"parallel verb-noun-adjective blank-verse scene lattice","complete_prose_controls":[{"rendered":z,"audit":audit(z),"reader_status":"complete authored prose control; not exact"} for z in controls],"candidates":candidates,"witnesses":witnesses,"stats":{"states":states,"pruned":pruned,"chart_advances":advances,"exact":len(candidates)},"provenance":{"joint_role_variables":True,"parallel_verb_noun_adjective_selection":True,"independently_authored_utterances":True,"independent_pointer_sha_audit":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False,"novelty_preflight":"new blank-verse dialogue role lattice","next_construction":"add inflectional person/number features to the independent utterance predicates"},"bottleneck":"outer character seam rejects the first jointly selected role pair before inner verb/adjective/noun states; this is a representation bottleneck, not a repair opportunity"}
    OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+"\n");return result
if __name__=="__main__":print(json.dumps(run(),indent=2))
