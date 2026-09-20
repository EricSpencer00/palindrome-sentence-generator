"""Live lexical equations with agreement-carrying two-word subject chunks."""
from __future__ import annotations

import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs"/"agreement-subject-chunk-equations-20260920.json"
EXPERIMENT_ID="agreement-subject-chunk-equations-20260920"
SIGNATURE="two-word-subject-chunk|agreement-carry|independent-boundaries|live-equation"

def letters(s:str)->str:return re.sub(r"[^a-z]","",s.casefold())
def audit(s:str)->dict[str,object]:
    t=letters(s);bad=[(i,len(t)-i-1) for i in range(len(t)//2) if t[i]!=t[-i-1]];f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters":len(t),"exact":bool(t) and not bad,"first_mismatch":bad[0] if bad else None,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}
def consume(l:str,r:str)->tuple[str,str]|None:
    n=min(len(l),len(r))
    if n and l[:n]!=r[-n:][::-1]:return None
    return l[n:],r[:-n] if n else r

@dataclass(frozen=True)
class Chunk:
    role:str;text:str;referent:str;valency:str;number:str|None=None
def chunks(role,ref,valency,*texts,number=None):return tuple(Chunk(role,t,ref,valency,number) for t in texts)

def banks():
    return (
        chunks("subject","agent1","animate-agent","the poet","a sailor","the guard","a reader",number="singular")
        + chunks("subject","agent1","animate-agent","the poets","some sailors","the guards","some readers",number="plural"),
        chunks("verb","event1","transitive","reads","opens","keeps","marks","guards"),
        chunks("object","theme1","inanimate-patient","the letter","the gate","the book","a message","the seal"),
        chunks("connector","relation","coordination","and","while","but"),
        chunks("subject","agent2","animate-agent","the queen","a captain","the poet","a friend",number="singular")
        + chunks("subject","agent2","animate-agent","the queens","some captains","the poets","some friends",number="plural"),
        chunks("verb","event2","transitive","reads","opens","keeps","marks","guards"),
        chunks("object","theme2","inanimate-patient","the letter","the gate","the book","a message","the seal"),
    )

def verb_agreement_ok(subject:Chunk,verb:Chunk)->bool:
    singular=verb.text.endswith("s")
    return (subject.number=="singular") == singular

def controls():
    texts=[
        "The poet reads the letter and the queen opens the gate.","A sailor opens the book while some guards mark the seal.",
        "The guard keeps a message but the poet reads the letter.","A reader marks the gate and the captain opens the book.",
        "The poets guard the book while a friend reads the message.","Some sailors open the gate and the queens mark the seal.",
        "The guards keep the letter but the poet opens the door.","Some readers mark the book while a captain reads the message.",
        "The poet opens the gate and some friends keep the book.","A sailor reads the seal while the guards mark the letter.",
        "The guard marks the message but the poets open the book.","The readers keep the gate and a queen reads the letter.",
        "The captain opens the book while some poets guard the seal.","A friend marks the letter and the guards keep the gate.",
        "The queens read the message but a sailor opens the book.","Some captains guard the letter while the poet marks the seal.",
        "The poet keeps the book and some readers open the gate.","A captain reads the letter but the guards mark the message.",
        "The sailors guard the seal while a queen opens the book.","Some friends read the gate and the poet keeps the letter.",
    ]
    return [{"rendered":t,"audit":audit(t),"reader_status":"complete contemporary prose control; not exact"} for t in texts]

def run(*,state_limit=250_000):
    lattice=banks();states=pruned=advances=feature_pruned=equations=0;survivors=[];rows=[]
    def walk(lo,hi,left,right,path,eq,subjects):
        nonlocal states,pruned,advances,feature_pruned,equations
        if states>=state_limit:return
        if lo>hi:
            if left or right or len(subjects)!=2:return
            if not verb_agreement_ok(subjects[0],path[1]) or not verb_agreement_ok(subjects[1],path[5]):feature_pruned+=1;return
            equations+=1;rendered=" ".join(x.text for x in path)+".";checked=audit(rendered);row={"rendered":rendered,"audit":checked,"equations":eq,"provenance":{"construction":"agreement-carrying two-word subject chunk equations","roles":[x.role for x in path],"numbers":[x.number for x in path],"referents":[x.referent for x in path],"independent_phrase_boundaries":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False},"reader_status":"unreviewed; exactness does not certify readability"};rows.append(row)
            if checked["exact"]:survivors.append(row)
            return
        if lo==hi:
            for x in lattice[lo]:
                states+=1;res=consume(left+letters(x.text),right)
                if res is None:pruned+=1;continue
                advances+=1;walk(lo+1,hi-1,res[0],res[1],path+(x,),eq+({"left_role":x.role,"left_text":x.text},),subjects)
            return
        for a in lattice[lo]:
            for b in lattice[hi]:
                states+=1
                # The left subject is available before the left verb.  The
                # right subject is deliberately still an open stack item, so
                # its verb agreement is checked at complete closure rather
                # than guessed early.
                if a.role=="verb" and subjects and not verb_agreement_ok(subjects[0],a):feature_pruned+=1;continue
                res=consume(left+letters(a.text),letters(b.text)+right)
                if res is None:pruned+=1;continue
                new_subjects=subjects
                if a.role=="subject":new_subjects=(a,subjects[1] if len(subjects)>1 else a)
                if b.role=="subject":new_subjects=(subjects[0] if subjects else b,b)
                advances+=1;walk(lo+1,hi-1,res[0],res[1],path+(a,),eq+({"left_role":a.role,"right_role":b.role,"left_text":a.text,"right_text":b.text,"left_letters":letters(a.text),"right_letters":letters(b.text)},),new_subjects)
    walk(0,len(lattice)-1,"","",(),(),())
    survivors.sort(key=lambda x:x["audit"]["letters"],reverse=True)
    result={"experiment":EXPERIMENT_ID,"method":"agreement-carrying two-word subject chunk equations","complete_prose_controls":controls(),"candidates":survivors,"equation_rows":rows[:200],"stats":{"states":states,"pruned":pruned,"feature_pruned":feature_pruned,"advances":advances,"equation_completions":equations,"exact":len(survivors)},"provenance":{"novelty_signature":SIGNATURE,"novelty_preflight":"fresh signature; two-word subject chunks carry number into verb selection before character output","two_word_subject_chunks":True,"agreement_before_emission":True,"independent_pointer_sha_audit":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False,"next_construction":"add a two-word object chunk with animacy/number selection while preserving subject agreement","reader_next_test":"blind all 20 controls against word-shuffled controls before promoting an exact row"}}
    OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+"\n");return result
if __name__=="__main__":print(json.dumps(run(),indent=2))
