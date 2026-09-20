"""Mixed singular/plural anaphora with explicit agreement branching."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs"/"mixed-agreement-anaphora-equations-20260920.json";EXPERIMENT_ID="mixed-agreement-anaphora-equations-20260920";SIGNATURE="mixed-number-branching|joint-anaphora|relation-conditioned-dative|live-equation"
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
def c(role,ref,valency,*texts,number=None,animacy=None,relation=None,anaphoric=False):return tuple(Chunk(role,t,ref,valency,number,animacy,relation,anaphoric) for t in texts)
def banks(state):
 sub=c("subject","mara","agent","Mara",number="singular",animacy="animate")+c("subject","noah","agent","Noah",number="singular",animacy="animate")+c("subject","guards","agent","the guards",number="plural",animacy="animate")
 verb=c("verb","event","dative","gives","sends","shows","brings","offers")
 theme=c("theme","theme","patient","the letter","a book","the seal",number="singular",animacy="inanimate")+c("theme","theme","patient","the letters","some books","the seals",number="plural",animacy="inanimate")
 prep=c("preposition","relation","recipient-preposition","for" if state=="benefit" else "to",relation=state)
 rec=c("recipient","child","animate-recipient","the child",number="singular",animacy="animate")+c("recipient","friend","animate-recipient","a friend",number="singular",animacy="animate")+c("recipient","guards","animate-recipient","the guards",number="plural",animacy="animate")
 conn=c("connector","relation","contrastive","although" if state=="benefit" else "while",relation=state)
 sub=c("subject","mara","anaphor","she",number="singular",animacy="animate",anaphoric=True)+c("subject","noah","anaphor","he",number="singular",animacy="animate",anaphoric=True)+c("subject","guards","anaphor","they",number="plural",animacy="animate",anaphoric=True)
 rec2=c("recipient","child","anaphoric-recipient","him",number="singular",animacy="animate",anaphoric=True)+c("recipient","friend","anaphoric-recipient","her",number="singular",animacy="animate",anaphoric=True)+c("recipient","guards","anaphoric-recipient","them",number="plural",animacy="animate",anaphoric=True)
 return sub,verb,theme,prep,rec,conn,sub,verb,theme,prep,rec2
def vok(s,v):return (s.number=="singular")==v.text.endswith("s")
def fok(s,v,t,p,r,cx,state):return vok(s,v) and v.valency=="dative" and t.valency=="patient" and t.animacy=="inanimate" and p.relation==state and p.text==("for" if state=="benefit" else "to") and r.animacy=="animate" and r.valency in {"animate-recipient","anaphoric-recipient"} and cx.relation==state
def controls():
 ts=["Mara gives the letter for the child although she sends a book for him.","Noah brings the seal for a friend although he offers books for her.","The guards give the letters for the guards although they send books for them.","Mara sends the letters for the child although she brings a book for him.","The guards give a book to the guards while they send the letters to them.","Noah brings the letter to a friend while he offers books to her.","The guards send the seal to a friend while they give a book to her.","Mara offers a letter to the guards while she shows the book to them.","The guards give books to the child while they send the seal to him.","Mara brings the letter to the guards while she shows a book to them.","Noah sends the letters to a friend while he gives a book to her.","The guards offer a book to the child while they give letters to him.","Mara sends the letter to a friend while she brings books to her.","The guards show the seal to the guards while they give a book to them.","Noah offers books to the guards while he sends the letter to them.","The guards bring a book to a friend while they give the seal to her.","Mara gives the book for a friend although she sends the seal for her.","The guards bring the letter to the child while they offer the book to him.","Noah shows seals to the guards while he gives a book to them.","The guards send books for a friend although they bring the letter for her."]
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
    if rs[0].ref!=ls[0].ref or not rs[0].anaphoric or rs[4].ref!=ls[4].ref or not rs[4].anaphoric or not fok(ls[0],ls[1],ls[2],ls[3],ls[4],ls[5],state) or not fok(rs[0],rs[1],rs[2],rs[3],rs[4],ls[5],state):feature_pruned+=1;return
    equations+=1;surface=li+ri;z=" ".join(x.text for x in surface)+".";a=audit(z);row={"rendered":z,"audit":a,"relation":state,"subject_number":ls[0].number,"recipient_number":ls[4].number,"subject_anaphor":rs[0].text,"recipient_anaphor":rs[4].text,"equations":eq,"provenance":{"construction":"mixed number agreement anaphora equations","roles":[x.role for x in surface],"referents":[x.ref for x in surface],"numbers":[x.number for x in surface],"anaphoric_flags":[x.anaphoric for x in surface],"relations":[x.relation for x in surface],"independent_phrase_boundaries":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False},"reader_status":"unreviewed; exactness does not certify readability"};rows.append(row)
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
 survivors.sort(key=lambda x:x["audit"]["letters"],reverse=True);result={"experiment":EXPERIMENT_ID,"method":"mixed singular/plural anaphora with relation equations","complete_prose_controls":controls(),"candidates":survivors,"equation_rows":rows[:200],"stats":{"relations":2,"states":states,"pruned":pruned,"feature_pruned":feature_pruned,"advances":advances,"equation_completions":equations,"exact":len(survivors)},"provenance":{"novelty_signature":SIGNATURE,"novelty_preflight":"fresh mixed-number signature; singular/plural branches are explicit closure states, not a bank expansion","mixed_agreement_branching":True,"joint_anaphoric_subject_recipient":True,"relation_state_retained":True,"recipient_theme_gate":True,"independent_pointer_sha_audit":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False,"next_construction":"add cross-clause number mismatch alternatives with explicit non-coreferential subjects","reader_next_test":"blind all 20 controls against word-shuffled controls before promoting any exact row"}}
 OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+"\n");return result
if __name__=="__main__":print(json.dumps(run(),indent=2))
