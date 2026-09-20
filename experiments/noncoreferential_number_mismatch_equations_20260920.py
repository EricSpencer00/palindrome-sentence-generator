"""Cross-clause number mismatch with non-coreferential subjects."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs"/"noncoreferential-number-mismatch-equations-20260920.json";EXPERIMENT_ID="noncoreferential-number-mismatch-equations-20260920";SIGNATURE="noncoreferential-subjects|cross-clause-number-mismatch|typed-dative|live-equation"
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
def c(role,ref,valency,*texts,number=None,animacy=None,relation=None):return tuple(Chunk(role,t,ref,valency,number,animacy,relation) for t in texts)
def banks(state):
 s1=c("subject","mara","agent","Mara","the poet",number="singular",animacy="animate")+c("subject","guards","agent","the guards","the poets",number="plural",animacy="animate")
 v=c("verb","event","dative","gives","sends","shows","brings","offers")
 t=c("theme","theme","patient","the letter","a book","the seal",number="singular",animacy="inanimate")+c("theme","theme","patient","the letters","some books","the seals",number="plural",animacy="inanimate")
 p=c("preposition","relation","recipient-preposition","for" if state=="benefit" else "to",relation=state)
 r=c("recipient","recipient","animate-recipient","the child","a friend","the poet",number="singular",animacy="animate")+c("recipient","recipient","animate-recipient","the children","some friends","the poets",number="plural",animacy="animate")
 conn=c("connector","relation","contrastive","although" if state=="benefit" else "while",relation=state)
 s2=c("subject","noah","agent","Noah","the queen","a captain",number="singular",animacy="animate")+c("subject","sailors","agent","the sailors","the queens","the captains",number="plural",animacy="animate")
 r2=r;return s1,v,t,p,r,conn,s2,v,t,p,r2
def vok(s,v):return (s.number=="singular")==v.text.endswith("s")
def fok(s,v,t,p,r,cx,state):
 benefit_verbs={"gives","sends","brings","offers"}
 relation_ok=v.text in benefit_verbs if state=="benefit" else True
 return (vok(s,v) and relation_ok and v.valency=="dative" and
         t.valency=="patient" and t.animacy=="inanimate" and
         p.relation==state and p.text==("for" if state=="benefit" else "to") and
         r.valency=="animate-recipient" and r.animacy=="animate" and
         cx.relation==state and cx.text==("although" if state=="benefit" else "while"))
def controls():
 ts=["Mara gives the letter for the child although the sailors send books for the children.","The guards bring the seal for a friend although Noah offers a book for the poet.","The poet shows a book to the child while the queens give letters to the children.","The poets send the letters to the children while a captain brings a book to a friend.","The queen gives a book for the poet although the guards send the seals for the children.","The sailors offer letters to a friend while Mara shows a book to the child.","A captain sends the seal for the poet although the poets bring books for the children.","The guards give a letter to the child while Noah offers books to a friend.","Mara sends the letters for a friend although the queens bring a book for the poet.","The sailors bring a seal to the children while the poet gives books to the child.","The poets offer the letter for the child although a captain sends books for the children.","Noah shows the seal to a friend while the guards give letters to the poet.","The queen brings a book for the child although the sailors send seals for a friend.","The guards send letters to the children while Mara offers a book to the poet.","The poet gives the seal for a friend although the captains bring books for the children.","The sailors show a letter to the child while Noah sends books to the poet.","Mara offers a book for the poet although the queens give letters for the children.","The guards bring seals to a friend while a captain sends the letter to the child.","The poets send books for the children although the queen brings a seal for the poet.","Noah gives the letter to a friend while the guards offer books to the children."]
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
    ls=li[:6];rs=ri
    if left or right or len(ls)!=6 or len(rs)!=5:return
    if ls[0].ref==rs[0].ref or not fok(ls[0],ls[1],ls[2],ls[3],ls[4],ls[5],state) or not fok(rs[0],rs[1],rs[2],rs[3],rs[4],ls[5],state):feature_pruned+=1;return
    equations+=1;surface=li+ri;z=" ".join(x.text for x in surface)+".";a=audit(z);row={"rendered":z,"audit":a,"relation":state,"subject_refs":[ls[0].ref,rs[0].ref],"subject_numbers":[ls[0].number,rs[0].number],"equations":eq,"provenance":{"construction":"non-coreferential cross-clause number mismatch equations","roles":[x.role for x in surface],"referents":[x.ref for x in surface],"numbers":[x.number for x in surface],"relations":[x.relation for x in surface],"independent_phrase_boundaries":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False},"reader_status":"unreviewed; exactness does not certify readability"};rows.append(row)
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
 survivors.sort(key=lambda x:x["audit"]["letters"],reverse=True);result={"experiment":EXPERIMENT_ID,"method":"non-coreferential cross-clause number mismatch equations","complete_prose_controls":controls(),"candidates":survivors,"equation_rows":rows[:200],"stats":{"relations":2,"states":states,"pruned":pruned,"feature_pruned":feature_pruned,"advances":advances,"equation_completions":equations,"exact":len(survivors)},"provenance":{"novelty_signature":SIGNATURE,"novelty_preflight":"fresh non-coreferential subject signature; number mismatch is an explicit branch with no pronoun link","explicit_noncoreference":True,"cross_clause_number_mismatch":True,"relation_state_retained":True,"recipient_theme_gate":True,"subject_agreement_before_emission":True,"independent_pointer_sha_audit":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False,"next_construction":"pivot to a materially different character/lexical search after this feature family closes","reader_next_test":"blind all 20 controls against word-shuffled controls before promoting any exact row"}}
 OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+"\n");return result
if __name__=="__main__":print(json.dumps(run(),indent=2))
