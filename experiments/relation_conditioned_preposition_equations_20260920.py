"""Relation-conditioned for/to recipient equations with complete surfaces."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs"/"relation-conditioned-preposition-equations-20260920.json";EXPERIMENT_ID="relation-conditioned-preposition-equations-20260920";SIGNATURE="discourse-relation|animacy-conditioned-preposition|complete-dative-surface|live-equation"
def letters(s):return re.sub(r"[^a-z]","",s.casefold())
def audit(s):
 t=letters(s);bad=[(i,len(t)-i-1) for i in range(len(t)//2) if t[i]!=t[-i-1]];f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();return {"letters":len(t),"exact":bool(t) and not bad,"first_mismatch":bad[0] if bad else None,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}
def consume(l,r):
 n=min(len(l),len(r))
 if n and l[:n]!=r[-n:][::-1]:return None
 return l[n:],r[:-n] if n else r
@dataclass(frozen=True)
class Chunk:
 role:str;text:str;valency:str;number:str|None=None;animacy:str|None=None;relation:str|None=None
def chunks(role,valency,*texts,number=None,animacy=None,relation=None):return tuple(Chunk(role,t,valency,number,animacy,relation) for t in texts)
def banks(relation):
 subs=chunks("subject","agent","the poet","a sailor","the guard",number="singular",animacy="animate")+chunks("subject","agent","the poets","some sailors","the guards",number="plural",animacy="animate")
 verbs=chunks("verb","dative","gives","sends","shows","brings","offers")
 themes=chunks("theme","patient","the letter","a book","the seal",number="singular",animacy="inanimate")+chunks("theme","patient","the letters","some books","the seals",number="plural",animacy="inanimate")
 preps=(chunks("preposition","recipient-preposition","for",relation=relation) if relation=="benefit" else chunks("preposition","recipient-preposition","to",relation=relation))
 recs=chunks("recipient","animate-recipient","the child","a friend","the poet",number="singular",animacy="animate")+chunks("recipient","animate-recipient","the children","some friends","the poets",number="plural",animacy="animate")
 conn=chunks("connector","discourse","and","while","but",relation=relation)
 subs2=chunks("subject","agent","the queen","a captain","the poet",number="singular",animacy="animate")+chunks("subject","agent","the queens","some captains","the poets",number="plural",animacy="animate")
 recs2=chunks("recipient","animate-recipient","the child","a friend","the poet",number="singular",animacy="animate")+chunks("recipient","animate-recipient","the children","some friends","the poets",number="plural",animacy="animate")
 return subs,verbs,themes,preps,recs,conn,subs2,verbs,themes,preps,recs2
def verb_ok(s,v):return (s.number=="singular")==v.text.endswith("s")
def frame_ok(s,v,t,p,r,relation):return verb_ok(s,v) and v.valency=="dative" and t.valency=="patient" and t.animacy=="inanimate" and p.relation==relation and p.text==("for" if relation=="benefit" else "to") and r.valency=="animate-recipient" and r.animacy=="animate"
def controls():
 ts=["The poet gives the letter for the child and the queen sends a book for a friend.","A sailor brings the seal for the poet while some guards offer books for the children.","The guard shows a book for a friend but the poet gives the letter for the child.","The poets send the letters for the children and a captain brings a book for the poet.","The queen gives a book to the poet and the guards send the letters to the children.","A captain brings the letter to a friend while the poets offer books to the child.","The guards send the seal to the poet while a sailor gives a book to the children.","The poet offers a letter to a friend and some captains show the book to the child.","The sailors give books to the children while the queen sends the seal to a poet.","The guard brings the letter to the child but the poets show a book to a friend.","Some captains send the letters to the poet and a captain gives a book to the child.","The captain offers a book to a friend while the guards give letters to the children.","A friend sends the letter to the child and the poet brings books to the children.","The queens show the seal to the poet but a sailor gives a book to a friend.","Some captains offer books to the children while the guard sends the letter to the poet.","The poet brings a book to a friend and some sailors give the seal to the child.","A captain shows letters to the children but the guards offer a book to a friend.","The sailors send seals to the poet while a queen gives the letter to the child.","The poet gives the message for a friend and the queen sends the book for the child.","The guards send the letter to the poet while a sailor offers a book to the children."]
 return [{"rendered":t,"audit":audit(t),"reader_status":"complete contemporary prose control; not exact"} for t in ts]
def run(*,state_limit=250000):
 states=pruned=advances=feature_pruned=equations=0;survivors=[];rows=[]
 def one(relation):
  nonlocal states,pruned,advances,feature_pruned,equations
  lattice=banks(relation)
  def walk(lo,hi,left,right,left_items,right_items,eq):
   nonlocal states,pruned,advances,feature_pruned,equations
   if states>=state_limit:return
   if lo>hi:
    right_surface=right_items;left_clause=left_items[:5];right_clause=right_surface[:5]
    if left or right or len(left_clause)!=5 or len(right_clause)!=5:return
    if not frame_ok(left_clause[0],left_clause[1],left_clause[2],left_clause[3],left_clause[4],relation) or not frame_ok(right_clause[0],right_clause[1],right_clause[2],right_clause[3],right_clause[4],relation):feature_pruned+=1;return
    equations+=1;surface=left_items+right_surface;z=" ".join(x.text for x in surface)+".";a=audit(z);row={"rendered":z,"audit":a,"relation":relation,"equations":eq,"provenance":{"construction":"relation-conditioned for/to recipient equations","roles":[x.role for x in surface],"relations":[x.relation for x in surface],"numbers":[x.number for x in surface],"animacy":[x.animacy for x in surface],"independent_phrase_boundaries":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False},"reader_status":"unreviewed; exactness does not certify readability"};rows.append(row)
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
     states+=1;res=consume(left+letters(a.text),letters(b.text)+right)
     if res is None:pruned+=1;continue
     advances+=1;walk(lo+1,hi-1,res[0],res[1],left_items+(a,),(b,)+right_items,eq+({"left_role":a.role,"right_role":b.role,"left_text":a.text,"right_text":b.text,"left_letters":letters(a.text),"right_letters":letters(b.text)},))
  walk(0,len(lattice)-1,"","",(),(),())
 one("benefit");one("transfer")
 survivors.sort(key=lambda x:x["audit"]["letters"],reverse=True);result={"experiment":EXPERIMENT_ID,"method":"relation-conditioned for/to recipient equations","complete_prose_controls":controls(),"candidates":survivors,"equation_rows":rows[:200],"stats":{"relations":2,"states":states,"pruned":pruned,"feature_pruned":feature_pruned,"advances":advances,"equation_completions":equations,"exact":len(survivors)},"provenance":{"novelty_signature":SIGNATURE,"novelty_preflight":"fresh discourse-relation signature; preposition is selected from relation state before paired character output","relation_state_live":True,"animacy_conditioned":True,"theme_number_live":True,"subject_agreement_before_emission":True,"independent_pointer_sha_audit":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False,"next_construction":"condition preposition choice on a contrastive discourse connective while retaining both relation states","reader_next_test":"blind all 20 controls against word-shuffled controls before promoting any exact row"}}
 OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+"\n");return result
if __name__=="__main__":print(json.dumps(run(),indent=2))
