"""Fresh two-clause construction from live reverse-compatible phrase equations."""
from __future__ import annotations

import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs"/"cross-word-equation-scene-20260920.json"
EXPERIMENT_ID="cross-word-equation-scene-20260920"


def letters(text:str)->str:return re.sub(r"[^a-z]","",text.casefold())
def audit(text:str)->dict[str,object]:
    t=letters(text); bad=[(i,len(t)-i-1) for i in range(len(t)//2) if t[i]!=t[-i-1]]; f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters":len(t),"exact":bool(t) and not bad,"first_mismatch":bad[0] if bad else None,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}
def consume(left:str,right:str)->tuple[str,str]|None:
    n=min(len(left),len(right))
    if n and left[:n]!=right[-n:][::-1]:return None
    return left[n:],right[:-n] if n else right

@dataclass(frozen=True)
class Chunk:
    role:str;text:str;referent:str;valency:str
def chunks(role,referent,valency,*texts):return tuple(Chunk(role,t,referent,valency) for t in texts)

def banks():
    # Complete ordinary two-clause frame:
    # subject verb object; connector; subject verb object.
    return (
        chunks("subject","agent1","animate-agent","Mara","Noah","the reader","the keeper","the sailor"),
        chunks("verb","event1","transitive","reads","opens","keeps","guards","marks"),
        chunks("object","theme1","inanimate-patient","the letter","the gate","the book","a message","the seal"),
        chunks("connector","relation","coordination","and","while","but"),
        chunks("subject","agent2","animate-agent","the queen","the guard","the poet","the captain","a friend"),
        chunks("verb","event2","transitive","reads","opens","keeps","guards","marks"),
        chunks("object","theme2","inanimate-patient","the letter","the gate","the book","a message","the seal"),
    )

def semantic_ok(path:tuple[Chunk,...])->bool:
    if len(path)!=7:return False
    if path[0].referent==path[4].referent:return False
    if path[1].valency!="transitive" or path[5].valency!="transitive":return False
    if path[2].referent==path[6].referent:return False
    return path[3].text in {"and","while","but"}

def controls():
    texts=[
        "Mara reads the letter and the queen opens the gate.","Noah opens the book while the guard marks the seal.",
        "The reader keeps a message but the poet reads the letter.","The keeper guards the gate and the captain opens the book.",
        "The sailor marks the seal while a friend keeps the message.","Mara opens the book and the poet guards the gate.",
        "Noah reads the seal while the queen keeps the letter.","The reader marks the gate but the guard opens the book.",
        "The keeper reads a message and the captain guards the seal.","The sailor keeps the book while the poet marks the letter.",
        "Mara guards the gate but a friend reads the message.","Noah marks the book and the queen keeps the seal.",
        "The reader opens the letter while the guard reads the gate.","The keeper keeps the message but the poet opens the book.",
        "The sailor reads the letter and the captain marks the gate.","Mara keeps the seal while the queen guards the book.",
        "Noah guards the message but the friend opens the gate.","The reader reads the book and the poet keeps the letter.",
        "The keeper marks the gate while the captain reads the seal.","The sailor opens the message and the guard guards the book.",
    ]
    return [{"rendered":t,"audit":audit(t),"reader_status":"complete contemporary prose control; not exact"} for t in texts]

def run(*,state_limit=250_000):
    lattice=banks(); states=pruned=advances=semantic_pruned=0; survivors=[]; witnesses=[]; equation_rows=[]
    # The center connector is selected as a lexical equation too.  Outer
    # chunks are selected from opposite semantic roles and may cross word
    # boundaries; no finished sentence exists during this search.
    def walk(lo,hi,left,right,path,eq):
        nonlocal states,pruned,advances,semantic_pruned
        if states>=state_limit:return
        if lo>hi:
            if left or right or not semantic_ok(path):semantic_pruned+=1;return
            rendered=" ".join(x.text for x in path)+".";checked=audit(rendered)
            row={"rendered":rendered,"audit":checked,"equations":eq,"provenance":{"construction":"cross-word reverse-compatible semantic scene equations","roles":[x.role for x in path],"referents":[x.referent for x in path],"valencies":[x.valency for x in path],"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False},"reader_status":"unreviewed; exactness does not certify readability"}
            if checked["exact"]:survivors.append(row)
            equation_rows.append({"rendered":rendered,"exact":checked["exact"],"letters":checked["letters"],"equation_steps":len(eq)})
            return
        if lo==hi:
            for x in lattice[lo]:
                states+=1;res=consume(left+letters(x.text),right)
                if res is None:pruned+=1;continue
                advances+=1;walk(lo+1,hi-1,res[0],res[1],path+(x,),eq+({"left_role":x.role,"right_role":None,"left_text":x.text,"right_text":None},))
            return
        for a in lattice[lo]:
            for b in lattice[hi]:
                states+=1
                if a.text==b.text or letters(a.text)==letters(a.text)[::-1] or letters(b.text)==letters(b.text)[::-1]:
                    semantic_pruned+=1;continue
                res=consume(left+letters(a.text),letters(b.text)+right)
                if res is None:pruned+=1;continue
                advances+=1;walk(lo+1,hi-1,res[0],res[1],path+(a,),eq+({"left_role":a.role,"right_role":b.role,"left_text":a.text,"right_text":b.text,"left_letters":letters(a.text),"right_letters":letters(b.text)},))
    walk(0,len(lattice)-1,"","",(),())
    survivors.sort(key=lambda x:x["audit"]["letters"],reverse=True)
    result={"experiment":EXPERIMENT_ID,"method":"live cross-word lexical equations over complete semantic scene","complete_prose_controls":controls(),"candidates":survivors,"equation_rows":equation_rows[:200],"witnesses":witnesses,"stats":{"states":states,"pruned":pruned,"semantic_pruned":semantic_pruned,"advances":advances,"equation_completions":len(equation_rows),"exact":len(survivors)},"provenance":{"novelty_preflight":"fresh signature; starts from lexical equations over semantic roles rather than a finished palindrome or fixed seed","cross_word_boundaries":True,"strict_semantic_gate":True,"independent_pointer_sha_audit":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False,"next_construction":"permit two-word subject chunks with explicit agreement while retaining equation traces","reader_next_test":"blind the 20 intact controls against word-shuffled controls before promoting any exact row"}}
    OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+"\n");return result
if __name__=="__main__":print(json.dumps(run(),indent=2))
