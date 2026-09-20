"""Contrastive subordinate subject-shift equations with retained relation state."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs"/"contrastive-subject-shift-equations-20260920.json";EXPERIMENT_ID="contrastive-subject-shift-equations-20260920";SIGNATURE="contrastive-subject-shift|retained-recipient-relation|complete-dative-surface|live-equation"
def letters(s):return re.sub(r"[^a-z]","",s.casefold())
def audit(s):
 t=letters(s);bad=[(i,len(t)-i-1) for i in range(len(t)//2) if t[i]!=t[-i-1]];f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();return {"letters":len(t),"exact":bool(t) and not bad,"first_mismatch":bad[0] if bad else None,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}
def consume(l,r):
 n=min(len(l),len(r))
 if n and l[:n]!=r[-n:][::-1]:return None
 return l[n:],r[:-n] if n else r
@dataclass(frozen=True)
class Chunk:
 role:str;text:str;ref:str;valency:str;number:str|None=None;animacy:str|None=None;relation:str|None=None
def chunks(role,ref,valency,*texts,number=None,animacy=None,relation=None):return tuple(Chunk(role,t,ref,valency,number,animacy,relation) for t in texts)
def banks(state):
 s1=chunks("subject","mara","agent","Mara","the poet","a sailor",number="singular",animacy="animate")+chunks("subject","guards","agent","the guards","the poets",number="plural",animacy="animate")
 v=chunks("verb","event","dative","gives","sends","shows","brings","offers")
 t=chunks("theme","theme","patient","the letter","a book","the seal",number="singular",animacy="inanimate")+chunks("theme","theme","patient","the letters","some books","the seals",number="plural",animacy="inanimate")
 p=chunks("preposition","relation","recipient-preposition","for" if state=="benefit" else "to",relation=state)
 r=chunks("recipient","recipient","animate-recipient","the child","a friend","the poet",number="singular",animacy="animate")+chunks("recipient","recipient","animate-recipient","the children","some friends","the poets",number="plural",animacy="animate")
 c=chunks("connector","relation","contrastive","although" if state=="benefit" else "while",relation=state)
 s2=chunks("subject","noah","agent","Noah","the queen","a captain",number="singular",animacy="animate")+chunks("subject","sailors","agent","the sailors","the queens",number="plural",animacy="animate")
 return s1,v,t,p,r,c,s2,v,t,p,r
def vok(s,v):return (s.number=="singular")==v.text.endswith("s")
def fok(s,v,t,p,r,c,state):return vok(s,v) and v.valency=="dative" and t.valency=="patient" and t.animacy=="inanimate" and p.relation==state and p.text==("for" if state=="benefit" else "to") and r.valency=="animate-recipient" and r.animacy=="animate" and c.relation==state and c.text==("although" if state=="benefit" else "while")
def controls():
 ts=["Mara gives the letter for the child although Noah sends a book for a friend.","The poet brings the seal for the sailor although the queen offers books for the children.","A sailor shows a book for a friend although a captain gives the letter for the child.","The guards send the letters for the children although the poets bring a book for the poet.","Mara gives a book to the queen while Noah sends the letters to the children.","The poet brings the letter to a friend while the captain offers books to the child.","A sailor sends the seal to the poet while the queen gives a book to the children.","The guards offer letters to the poet while the captain shows a book to a friend.","Mara sends a letter for a friend although Noah brings the book for the child.","The poet gives books to the children while the queen sends the seal to a captain.","A sailor offers the letter to the child while Noah shows a book to the poet.","The guards bring the seals for the children although the queen gives a book for a friend.","Mara shows a book to the poet while a captain sends letters to the children.","The poets offer the letters for a friend although Noah brings a book for the child.","A sailor gives the seal to the queen while the guards send books to the poet.","The poet sends a book for the child although the captain gives letters for the children.","Mara brings the letter to a friend while Noah offers the book to the poet.","The guards show seals to the children while the queen gives a book to a friend.","A sailor offers books for the children although Noah sends the letter for a friend.","The poet brings a book to the child while the captain sends the seal to the queen."]
 return [{"rendered":t,"audit":audit(t),"reader_status":"complete contemporary prose control; not exact"} for t in ts]
def run(*,state_limit=250000):
 states=pruned=advances=feature_pruned=equations=0;survivors=[];rows=[]
 def one(state):
  nonlocal states,pruned,advances,feature_pruned,equations
  lattice=banks(state)
  def walk(lo,hi,left,right,li,ri,eq):
   nonlocal states,pruned,advances,feature_pruned,equations
   if states>=state_limit:return
   if lo>hi:
    rs=ri;ls=li[:6];right_clause=rs[:5]
    if left or right or len(ls)!=6 or len(right_clause)!=5:return
    if ls[0].ref==right_clause[0].ref:feature_pruned+=1;return
    if not fok(ls[0],ls[1],ls[2],ls[3],ls[4],ls[5],state) or not fok(right_clause[0],right_clause[1],right_clause[2],right_clause[3],right_clause[4],ls[5],state):feature_pruned+=1;return
    equations+=1;surface=li+ri;z=" ".join(x.text for x in surface)+".";a=audit(z);row={"rendered":z,"audit":a,"relation":state,"subject_shift":[ls[0].ref,right_clause[0].ref],"equations":eq,"provenance":{"construction":"contrastive subordinate subject-shift recipient equations","roles":[x.role for x in surface],"referents":[x.ref for x in surface],"relations":[x.relation for x in surface],"independent_phrase_boundaries":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False},"reader_status":"unreviewed; exactness does not certify readability"};rows.append(row)
    if a["exact"]:survivors.append(row)
    return
   if lo==hi:
    for x in lattice[lo]:
     states+=1;res=consume(left+letters(x.text),right)
     if res is None:pruned+=1;continue
     advances+=1;walk(lo+1,hi-1,res[0],res[1],li+(x,),ri,eq+({"left_role":x.role,"left_text":x.text},))
    return
   for a in lattice[lo]:
    for b in lattice[hi]:
     states+=1;res=consume(left+letters(a.text),letters(b.text)+right)
     if res is None:pruned+=1;continue
     advances+=1;walk(lo+1,hi-1,res[0],res[1],li+(a,),(b,)+ri,eq+({"left_role":a.role,"right_role":b.role,"left_text":a.text,"right_text":b.text,"left_letters":letters(a.text),"right_letters":letters(b.text)},))
  walk(0,len(lattice)-1,"","",(),(),())
 one("benefit");one("transfer")
 survivors.sort(key=lambda x:x["audit"]["letters"],reverse=True);result={"experiment":EXPERIMENT_ID,"method":"contrastive subject-shift recipient equations","complete_prose_controls":controls(),"candidates":survivors,"equation_rows":rows[:200],"stats":{"relations":2,"states":states,"pruned":pruned,"feature_pruned":feature_pruned,"advances":advances,"equation_completions":equations,"exact":len(survivors)},"provenance":{"novelty_signature":SIGNATURE,"novelty_preflight":"fresh subject-shift signature; distinct subject referents are retained through both complete clause states","explicit_subject_shift":True,"relation_state_retained":True,"recipient_theme_gate":True,"subject_agreement_before_emission":True,"independent_pointer_sha_audit":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False,"next_construction":"add anaphoric pronoun realization for the shifted subject while retaining explicit referent identity","reader_next_test":"blind all 20 controls against word-shuffled controls before promoting any exact row"}}
 OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+"\n");return result
if __name__=="__main__":print(json.dumps(run(),indent=2))
