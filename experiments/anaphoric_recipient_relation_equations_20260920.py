"""Anaphoric recipient pronouns with typed antecedents and live equations."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs"/"anaphoric-recipient-relation-equations-20260920.json";EXPERIMENT_ID="anaphoric-recipient-relation-equations-20260920";SIGNATURE="anaphoric-recipient|typed-antecedent|anaphoric-subject|relation-conditioned-dative|live-equation"
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
 rec=chunks("recipient","child","animate-recipient","the child",number="singular",animacy="animate")+chunks("recipient","friend","animate-recipient","a friend",number="singular",animacy="animate")+chunks("recipient","poet","animate-recipient","the poet",number="singular",animacy="animate")+chunks("recipient","guards","animate-recipient","the guards",number="plural",animacy="animate")
 c=chunks("connector","relation","contrastive","although" if state=="benefit" else "while",relation=state)
 s2=chunks("subject","mara","anaphor","she",number="singular",animacy="animate",anaphoric=True)+chunks("subject","noah","anaphor","he",number="singular",animacy="animate",anaphoric=True)+chunks("subject","guards","anaphor","they",number="plural",animacy="animate",anaphoric=True)
 rec2=chunks("recipient","child","anaphoric-recipient","him",number="singular",animacy="animate",anaphoric=True)+chunks("recipient","friend","anaphoric-recipient","her",number="singular",animacy="animate",anaphoric=True)+chunks("recipient","poet","anaphoric-recipient","them",number="singular",animacy="animate",anaphoric=True)+chunks("recipient","guards","anaphoric-recipient","them",number="plural",animacy="animate",anaphoric=True)
 return s1,v,t,p,rec,c,s2,v,t,p,rec2
def vok(s,v):return (s.number=="singular")==v.text.endswith("s")
def fok(s,v,t,p,r,c,state):
 benefit_verbs={"gives","sends","brings","offers"}
 relation_ok=(v.text in benefit_verbs) if state=="benefit" else True
 return (vok(s,v) and relation_ok and v.valency=="dative" and
         t.valency=="patient" and t.animacy=="inanimate" and
         p.relation==state and p.text==("for" if state=="benefit" else "to") and
         r.valency in {"animate-recipient","anaphoric-recipient"} and
         r.animacy=="animate" and c.relation==state and
         c.text==("although" if state=="benefit" else "while"))
def controls():
 ts=["Mara gives the letter for the child although she sends a book for him.","Noah brings the seal for a friend although he offers books for her.","The guards bring a book for a friend although they give the letter for her.","Mara sends the letters for the child although she brings a book for him.","Mara gives a book to the friend while she sends the letters to her.","Noah brings the letter to a friend while he offers books to her.","The guards send the seal to the child while they give a book to him.","Mara offers a letter to a friend while she shows the book to her.","Noah gives books to a friend while he sends the seal to her.","The guards bring the letter to the child while they show a book to him.","Mara sends the letters to a friend while she gives a book to her.","Noah offers a book to a friend while he gives letters to her.","The guards send the letter to the child while they bring books to him.","Mara shows the seal to a friend while she gives a book to her.","Noah offers books to a friend while he sends the letter to her.","The guards bring a book to the child while they give the seal to him.","Mara gives the book for a friend although she sends the seal for her.","Noah brings the letter to a friend while he offers the book to her.","The guards show seals to the child while they give a book to him.","Mara sends books for a friend although she brings the letter for her."]
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
    if rs[0].ref!=ls[0].ref or not rs[0].anaphoric or not rs[4].anaphoric or rs[4].ref!=ls[4].ref or rs[4].number!=ls[4].number or not fok(ls[0],ls[1],ls[2],ls[3],ls[4],ls[5],state) or not fok(rs[0],rs[1],rs[2],rs[3],rs[4],ls[5],state):feature_pruned+=1;return
    equations+=1;surface=li+ri;z=" ".join(x.text for x in surface)+".";a=audit(z);row={"rendered":z,"audit":a,"relation":state,"subject_antecedent":ls[0].ref,"subject_anaphor":rs[0].text,"recipient_antecedent":ls[4].ref,"recipient_anaphor":rs[4].text,"equations":eq,"provenance":{"construction":"anaphoric recipient relation equations","roles":[x.role for x in surface],"referents":[x.ref for x in surface],"anaphoric_flags":[x.anaphoric for x in surface],"relations":[x.relation for x in surface],"independent_phrase_boundaries":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False},"reader_status":"unreviewed; exactness does not certify readability"};rows.append(row)
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
 survivors.sort(key=lambda x:x["audit"]["letters"],reverse=True);result={"experiment":EXPERIMENT_ID,"method":"anaphoric recipient realization with relation-conditioned equations","complete_prose_controls":controls(),"candidates":survivors,"equation_rows":rows[:200],"stats":{"relations":2,"states":states,"pruned":pruned,"feature_pruned":feature_pruned,"advances":advances,"equation_completions":equations,"exact":len(survivors)},"provenance":{"novelty_signature":SIGNATURE,"novelty_preflight":"fresh anaphoric-recipient signature; pronoun identity is checked against the typed recipient antecedent","explicit_recipient_antecedent":True,"anaphoric_subject_retained":True,"anaphoric_recipient_live":True,"relation_state_retained":True,"recipient_theme_gate":True,"subject_agreement_before_emission":True,"independent_pointer_sha_audit":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False,"next_construction":"allow a jointly anaphoric subject and recipient with plural agreement and explicit coreference constraints","reader_next_test":"blind all 20 controls against word-shuffled controls before promoting any exact row"}}
 OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+"\n");return result
if __name__=="__main__":print(json.dumps(run(),indent=2))
