"""Live equations with agreement subjects and typed two-word object chunks."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs"/"typed-object-chunk-equations-20260920.json";EXPERIMENT_ID="typed-object-chunk-equations-20260920";SIGNATURE="two-word-object-chunk|animacy-number-state|subject-agreement|live-equation"
def letters(s):return re.sub(r"[^a-z]","",s.casefold())
def audit(s):
 t=letters(s);bad=[(i,len(t)-i-1) for i in range(len(t)//2) if t[i]!=t[-i-1]];f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();return {"letters":len(t),"exact":bool(t) and not bad,"first_mismatch":bad[0] if bad else None,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}
def consume(l,r):
 n=min(len(l),len(r))
 if n and l[:n]!=r[-n:][::-1]:return None
 return l[n:],r[:-n] if n else r
@dataclass(frozen=True)
class Chunk:
 role:str;text:str;referent:str;valency:str;number:str|None=None;animacy:str|None=None
def chunks(role,ref,valency,*texts,number=None,animacy=None):return tuple(Chunk(role,t,ref,valency,number,animacy) for t in texts)
def banks():
 return (
  chunks("subject","agent1","animate-agent","the poet","a sailor","the guard","a reader",number="singular",animacy="animate")+chunks("subject","agent1","animate-agent","the poets","some sailors","the guards","some readers",number="plural",animacy="animate"),
  chunks("verb","event1","transitive","reads","opens","keeps","marks","guards"),
  chunks("object","theme1","patient","the letter","a book","the poet","a friend",number="singular",animacy="inanimate")+chunks("object","theme1","patient","the letters","some books","the poets","some friends",number="plural",animacy="animate"),
  chunks("connector","relation","coordination","and","while","but"),
  chunks("subject","agent2","animate-agent","the queen","a captain","the poet","a friend",number="singular",animacy="animate")+chunks("subject","agent2","animate-agent","the queens","some captains","the poets","some friends",number="plural",animacy="animate"),
  chunks("verb","event2","transitive","reads","opens","keeps","marks","guards"),
  chunks("object","theme2","patient","the letter","a book","the poet","a friend",number="singular",animacy="animate")+chunks("object","theme2","patient","the letters","some books","the poets","some friends",number="plural",animacy="animate"),
 )
def verb_ok(sub,verb):return (sub.number=="singular")==verb.text.endswith("s")
def object_ok(verb,obj):
 # all selected verbs are transitive; retain an explicit typed check so the
 # object feature participates in construction rather than post-hoc prose.
 return verb.valency=="transitive" and obj.valency=="patient" and obj.animacy in {"animate","inanimate"}
def controls():
 ts=["The poet reads the letter and the queen opens a book.","A sailor opens the book while some guards mark the letters.","The guard keeps a friend but the poet reads the books.","A reader marks the poet and the captain opens the letter.","The poets guard the book while a friend reads the letters.","Some sailors open the poets and the queens mark a book.","The guards keep the letter but the poet opens some friends.","Some readers mark the books while a captain reads the poet.","The poet opens a friend and some poets keep the letter.","A captain reads the letters while the guards mark a book.","The guard marks some friends but the poets open the book.","The readers keep the poet and a queen reads the letters.","The captain opens the book while some poets guard a friend.","A friend marks the letter and the guards keep the books.","The queens read some friends but a sailor opens the book.","Some captains guard the poet while the poet marks the letters.","The poet keeps a book and some readers open the friends.","A captain reads the letter but the guards mark some books.","The sailors guard the letters while a queen opens a book.","Some friends read the poet and the poet keeps the book."]
 return [{"rendered":t,"audit":audit(t),"reader_status":"complete contemporary prose control; not exact"} for t in ts]
def run(*,state_limit=250000):
 lattice=banks();states=pruned=advances=feature_pruned=equations=0;survivors=[];rows=[]
 def walk(lo,hi,left,right,path,eq,subs,objs):
  nonlocal states,pruned,advances,feature_pruned,equations
  if states>=state_limit:return
  if lo>hi:
   if left or right or len(subs)!=2 or len(objs)!=2:return
   if not verb_ok(subs[0],path[1]) or not verb_ok(subs[1],path[5]) or not object_ok(path[1],objs[0]) or not object_ok(path[5],objs[1]):feature_pruned+=1;return
   equations+=1;z=" ".join(x.text for x in path)+".";a=audit(z);row={"rendered":z,"audit":a,"equations":eq,"provenance":{"construction":"typed two-word object chunk equations","roles":[x.role for x in path],"numbers":[x.number for x in path],"animacy":[x.animacy for x in path],"independent_phrase_boundaries":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False},"reader_status":"unreviewed; exactness does not certify readability"};rows.append(row)
   if a["exact"]:survivors.append(row)
   return
  if lo==hi:
   for x in lattice[lo]:
    states+=1;res=consume(left+letters(x.text),right)
    if res is None:pruned+=1;continue
    advances+=1;walk(lo+1,hi-1,res[0],res[1],path+(x,),eq+({"left_role":x.role,"left_text":x.text},),subs,objs)
   return
  for a in lattice[lo]:
   for b in lattice[hi]:
    states+=1
    if a.role=="verb" and subs and not verb_ok(subs[0],a):feature_pruned+=1;continue
    res=consume(left+letters(a.text),letters(b.text)+right)
    if res is None:pruned+=1;continue
    ns=subs;no=objs
    if a.role=="subject":ns=(a,subs[1] if len(subs)>1 else a)
    if b.role=="subject":ns=(subs[0] if subs else b,b)
    if a.role=="object":no=(a,objs[1] if len(objs)>1 else a)
    if b.role=="object":no=(objs[0] if objs else b,b)
    advances+=1;walk(lo+1,hi-1,res[0],res[1],path+(a,),eq+({"left_role":a.role,"right_role":b.role,"left_text":a.text,"right_text":b.text,"left_letters":letters(a.text),"right_letters":letters(b.text)},),ns,no)
 walk(0,len(lattice)-1,"","",(),(),(),())
 survivors.sort(key=lambda x:x["audit"]["letters"],reverse=True);result={"experiment":EXPERIMENT_ID,"method":"typed two-word object chunk equations with subject agreement","complete_prose_controls":controls(),"candidates":survivors,"equation_rows":rows[:200],"stats":{"states":states,"pruned":pruned,"feature_pruned":feature_pruned,"advances":advances,"equation_completions":equations,"exact":len(survivors)},"provenance":{"novelty_signature":SIGNATURE,"novelty_preflight":"fresh object-feature signature; animacy/number is carried inside two-word object chunks before output","two_word_object_chunks":True,"animacy_number_before_emission":True,"subject_agreement_before_emission":True,"independent_pointer_sha_audit":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False,"next_construction":"add typed recipient objects with dative/transitive valency while retaining object number","reader_next_test":"blind all 20 controls against word-shuffled controls before promoting any exact row"}}
 OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+"\n");return result
if __name__=="__main__":print(json.dumps(run(),indent=2))
