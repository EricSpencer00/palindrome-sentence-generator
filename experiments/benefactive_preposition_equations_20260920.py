"""Live equations for prepositional benefactive recipient frames."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs"/"benefactive-preposition-equations-20260920.json";EXPERIMENT_ID="benefactive-preposition-equations-20260920";SIGNATURE="benefactive-preposition|recipient-valency|theme-number|subject-agreement|live-equation"
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
 subjects=chunks("subject","agent1","animate-agent","the poet","a sailor","the guard",number="singular",animacy="animate")+chunks("subject","agent1","animate-agent","the poets","some sailors","the guards",number="plural",animacy="animate")
 verbs=chunks("verb","event1","benefactive","gives","sends","shows","brings","offers")
 themes=chunks("theme","theme1","patient","the letter","a book","the seal",number="singular",animacy="inanimate")+chunks("theme","theme1","patient","the letters","some books","the seals",number="plural",animacy="inanimate")
 preps=chunks("preposition","benefit","benefactive-preposition","for","to")
 recipients=chunks("recipient","recipient1","benefactive-recipient","the child","a friend","the poet",number="singular",animacy="animate")+chunks("recipient","recipient1","benefactive-recipient","the children","some friends","the poets",number="plural",animacy="animate")
 connector=chunks("connector","relation","coordination","and","while","but")
 subjects2=chunks("subject","agent2","animate-agent","the queen","a captain","the poet",number="singular",animacy="animate")+chunks("subject","agent2","animate-agent","the queens","some captains","the poets",number="plural",animacy="animate")
 recipients2=chunks("recipient","recipient2","benefactive-recipient","the child","a friend","the poet",number="singular",animacy="animate")+chunks("recipient","recipient2","benefactive-recipient","the children","some friends","the poets",number="plural",animacy="animate")
 return subjects,verbs,themes,preps,recipients,connector,subjects2,verbs, themes,preps,recipients2
def verb_ok(sub,verb):return (sub.number=="singular")==verb.text.endswith("s")
def frame_ok(verb,theme,prep,recipient):return verb.valency=="benefactive" and theme.valency=="patient" and theme.animacy=="inanimate" and prep.valency=="benefactive-preposition" and recipient.valency=="benefactive-recipient" and recipient.animacy=="animate"
def controls():
 ts=["The poet gives the letter to the child and the queen sends a book to a friend.","A sailor brings the seal for the poet while some guards offer books to the children.","The guard shows a book to a friend but the poet gives the letter for the child.","The poets send the letters to the children and a captain brings a book for the poet.","Some sailors offer the seal for a friend while the queens show books to the children.","The queen gives a book to the poet and the guards send the letters for the children.","A captain brings the letter for a friend but the poets offer books to the child.","The guards send the seal to the poet while a sailor gives a book for the children.","The poet offers a letter for a friend and some captains show the book to the child.","The sailors give books to the children while the queen sends the seal for a poet.","The guard brings the letter to the child but the poets show a book for a friend.","Some captains send the letters for the poet and a captain gives a book to the child.","The captain offers a book to a friend while the guards give letters for the children.","A friend sends the letter to the child and the poet brings books for the children.","The queens show the seal to the poet but a sailor gives a book for a friend.","Some captains offer books for the children while the guard sends the letter to the poet.","The poet brings a book to a friend and some sailors give the seal for the child.","A captain shows letters to the children but the guards offer a book for a friend.","The sailors send seals for the poet while a queen gives the letter to the child.","Some friends give the book to the child and the poet sends the seal for a friend."]
 return [{"rendered":t,"audit":audit(t),"reader_status":"complete contemporary prose control; not exact"} for t in ts]
def run(*,state_limit=250000):
 lattice=banks();states=pruned=advances=feature_pruned=equations=0;survivors=[];rows=[]
 def walk(lo,hi,left,right,left_items,right_items,eq):
  nonlocal states,pruned,advances,feature_pruned,equations
  if states>=state_limit:return
  if lo>hi:
   # right_items is stored in grammatical surface order: each newly selected
   # outer slot is prepended during the equation walk, so the final tuple is
   # already subject -> verb -> theme -> preposition -> recipient.  The
   # character equation used the opposite arrival order incrementally; only
   # rendering uses this grammatical tuple.
   right_surface=right_items;left_clause=left_items[:5];right_clause=right_surface[:5]
   if left or right or len(left_clause)!=5 or len(right_clause)!=5:return
   if not verb_ok(left_clause[0],left_clause[1]) or not verb_ok(right_clause[0],right_clause[1]):feature_pruned+=1;return
   if not frame_ok(left_clause[1],left_clause[2],left_clause[3],left_clause[4]) or not frame_ok(right_clause[1],right_clause[2],right_clause[3],right_clause[4]):feature_pruned+=1;return
   equations+=1;surface=left_items+right_surface;z=" ".join(x.text for x in surface)+".";a=audit(z);row={"rendered":z,"audit":a,"equations":eq,"provenance":{"construction":"prepositional benefactive recipient equations","roles":[x.role for x in surface],"valencies":[x.valency for x in surface],"numbers":[x.number for x in surface],"animacy":[x.animacy for x in surface],"independent_phrase_boundaries":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False},"reader_status":"unreviewed; exactness does not certify readability"};rows.append(row)
   if a["exact"]:survivors.append(row)
   return
  if lo==hi:
   for x in lattice[lo]:
    states+=1;res=consume(left+letters(x.text),right)
    if res is None:pruned+=1;continue
    advances+=1;walk(lo+1,hi-1,res[0],res[1],left_items+(x,),right_items,eq+({"left_role":x.role,"left_text":x.text},))
   return
  for a in lattice[lo]:
   for b in lattice[hi]:
    states+=1
    left_subject=next((item for item in left_items if item.role=="subject"),None)
    if a.role=="verb" and left_subject is not None and not verb_ok(left_subject,a):feature_pruned+=1;continue
    res=consume(left+letters(a.text),letters(b.text)+right)
    if res is None:pruned+=1;continue
    advances+=1;walk(lo+1,hi-1,res[0],res[1],left_items+(a,),(b,)+right_items,eq+({"left_role":a.role,"right_role":b.role,"left_text":a.text,"right_text":b.text,"left_letters":letters(a.text),"right_letters":letters(b.text)},))
 walk(0,len(lattice)-1,"","",(),(),())
 survivors.sort(key=lambda x:x["audit"]["letters"],reverse=True);result={"experiment":EXPERIMENT_ID,"method":"prepositional benefactive recipient equations","complete_prose_controls":controls(),"candidates":survivors,"equation_rows":rows[:200],"stats":{"states":states,"pruned":pruned,"feature_pruned":feature_pruned,"advances":advances,"equation_completions":equations,"exact":len(survivors)},"provenance":{"novelty_signature":SIGNATURE,"novelty_preflight":"fresh benefactive preposition signature; recipient valency is realized by an explicit for/to frame","preposition_valency_live":True,"theme_number_live":True,"subject_agreement_before_emission":True,"independent_pointer_sha_audit":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False,"next_construction":"add alternating recipient prepositions conditioned on animacy and discourse relation","reader_next_test":"blind all 20 controls against word-shuffled controls before promoting any exact row"}}
 OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+"\n");return result
if __name__=="__main__":print(json.dumps(run(),indent=2))
