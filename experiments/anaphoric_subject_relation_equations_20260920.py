"""Anaphoric shifted-subject realization with live relation equations."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs"/"anaphoric-subject-relation-equations-20260920.json";EXPERIMENT_ID="anaphoric-subject-relation-equations-20260920";SIGNATURE="anaphoric-subject|explicit-referent-identity|relation-conditioned-dative|live-equation"
def letters(s):return re.sub(r"[^a-z]","",s.casefold())
def audit(s):
 t=letters(s);bad=[(i,len(t)-i-1) for i in range(len(t)//2) if t[i]!=t[-i-1]];f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();return {"letters":len(t),"exact":bool(t) and not bad,"first_mismatch":bad[0] if bad else None,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}
def consume(l,r):
 n=min(len(l),len(r))
 if n and l[:n]!=r[-n:][::-1]:return None
 return l[n:],r[:-n] if n else r
@dataclass(frozen=True)
class Chunk:
 role:str;text:str;ref:str;valency:str;number:str|None=None;animacy:str|None=None;relation:str|None=None;anaphoric:bool=False
def chunks(role,ref,valency,*texts,number=None,animacy=None,relation=None,anaphoric=False):return tuple(Chunk(role,t,ref,valency,number,animacy,relation,anaphoric) for t in texts)
def banks(state):
 s1=chunks("subject","mara","agent","Mara",number="singular",animacy="animate")+chunks("subject","noah","agent","Noah",number="singular",animacy="animate")+chunks("subject","guards","agent","the guards",number="plural",animacy="animate")
 v=chunks("verb","event","dative","gives","sends","shows","brings","offers")
 t=chunks("theme","theme","patient","the letter","a book","the seal",number="singular",animacy="inanimate")+chunks("theme","theme","patient","the letters","some books","the seals",number="plural",animacy="inanimate")
 p=chunks("preposition","relation","recipient-preposition","for" if state=="benefit" else "to",relation=state)
 r=chunks("recipient","recipient","animate-recipient","the child","a friend","the poet",number="singular",animacy="animate")+chunks("recipient","recipient","animate-recipient","the children","some friends","the poets",number="plural",animacy="animate")
 c=chunks("connector","relation","contrastive","although" if state=="benefit" else "while",relation=state)
 pronouns=chunks("subject","mara","anaphor","she",number="singular",animacy="animate",anaphoric=True)+chunks("subject","noah","anaphor","he",number="singular",animacy="animate",anaphoric=True)+chunks("subject","guards","anaphor","they",number="plural",animacy="animate",anaphoric=True)
 return s1,v,t,p,r,c,pronouns,v,t,p,r
def vok(s,v):return (s.number=="singular")==v.text.endswith("s")
def fok(s,v,t,p,r,c,state):
 benefit_verbs={"gives","sends","brings","offers"}
 relation_ok=(v.text in benefit_verbs) if state=="benefit" else True
 return (vok(s,v) and relation_ok and v.valency=="dative" and
         t.valency=="patient" and t.animacy=="inanimate" and
         p.relation==state and p.text==("for" if state=="benefit" else "to") and
         r.valency=="animate-recipient" and r.animacy=="animate" and
         c.relation==state and c.text==("although" if state=="benefit" else "while"))
def controls():
 ts=["Mara gives the letter for the child although she sends a book for a friend.","Noah brings the seal for the poet although he offers books for the children.","The guards bring a book for a friend although they give the letter for the child.","Mara sends the letters for the children although she brings a book for the poet.","Mara gives a book to the poet while she sends the letters to the children.","Noah brings the letter to a friend while he offers books to the child.","The guards send the seal to the poet while they give a book to the children.","Mara offers a letter to a friend while she shows the book to the child.","Noah gives books to the children while he sends the seal to a poet.","The guards bring the letter to the child while they show a book to a friend.","Mara sends the letters to the poet while she gives a book to the child.","Noah offers a book to a friend while he gives letters to the children.","The guards send the letter to the child while they bring books to the children.","Mara shows the seal to the poet while she gives a book to a friend.","Noah offers books to the children while he sends the letter to the poet.","The guards bring a book to a friend while they give the seal to the child.","Mara gives the book for a friend although she sends the seal for the child.","Noah brings the letter to a friend while he offers the book to the poet.","The guards show seals to the children while they give a book to a friend.","Mara sends books for the children although she brings the letter for a friend."]
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
    if rs[0].ref!=ls[0].ref or not rs[0].anaphoric or not fok(ls[0],ls[1],ls[2],ls[3],ls[4],ls[5],state) or not fok(rs[0],rs[1],rs[2],rs[3],rs[4],ls[5],state):feature_pruned+=1;return
    equations+=1;surface=li+ri;z=" ".join(x.text for x in surface)+".";a=audit(z);row={"rendered":z,"audit":a,"relation":state,"antecedent":ls[0].ref,"anaphor":rs[0].text,"equations":eq,"provenance":{"construction":"anaphoric shifted subject relation equations","roles":[x.role for x in surface],"referents":[x.ref for x in surface],"anaphoric_flags":[x.anaphoric for x in surface],"relations":[x.relation for x in surface],"independent_phrase_boundaries":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False},"reader_status":"unreviewed; exactness does not certify readability"};rows.append(row)
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
 survivors.sort(key=lambda x:x["audit"]["letters"],reverse=True);result={"experiment":EXPERIMENT_ID,"method":"anaphoric subject realization with relation-conditioned recipient equations","complete_prose_controls":controls(),"candidates":survivors,"equation_rows":rows[:200],"stats":{"relations":2,"states":states,"pruned":pruned,"feature_pruned":feature_pruned,"advances":advances,"equation_completions":equations,"exact":len(survivors)},"provenance":{"novelty_signature":SIGNATURE,"novelty_preflight":"fresh anaphoric-subject signature; pronoun identity is checked against the antecedent before closure","explicit_referent_identity":True,"anaphoric_subject_live":True,"relation_state_retained":True,"recipient_theme_gate":True,"subject_agreement_before_emission":True,"independent_pointer_sha_audit":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False,"next_construction":"allow anaphoric recipient pronouns with explicit antecedent typing","reader_next_test":"blind all 20 controls against word-shuffled controls before promoting any exact row"}}
 OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+"\n");return result
if __name__=="__main__":print(json.dumps(run(),indent=2))
