"""Joint plural subject/recipient anaphora with relation equations."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs"/"joint-plural-anaphora-equations-20260920.json";EXPERIMENT_ID="joint-plural-anaphora-equations-20260920";SIGNATURE="joint-plural-anaphora|subject-recipient-coreference|relation-conditioned-dative|live-equation"
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
 # The surface deliberately keeps plural antecedents on the first clause and
 # plural pronouns on the second clause.
 s=c("subject","guards","agent","the guards",number="plural",animacy="animate")
 v=c("verb","event","dative","give","send","show","bring","offer")
 t=c("theme","theme","patient","the letters","some books","the seals",number="plural",animacy="inanimate")
 p=c("preposition","relation","recipient-preposition","for" if state=="benefit" else "to",relation=state)
 r=c("recipient","children","animate-recipient","the children",number="plural",animacy="animate")
 conn=c("connector","relation","contrastive","although" if state=="benefit" else "while",relation=state)
 s2=c("subject","guards","anaphor","they",number="plural",animacy="animate",anaphoric=True)
 r2=c("recipient","children","anaphoric-recipient","them",number="plural",animacy="animate",anaphoric=True)
 return s,v,t,p,r,conn,s2,v,t,p,r2
def fok(s,v,t,p,r,conn,state):return s.number=="plural" and v.valency=="dative" and not v.text.endswith("s") and t.number=="plural" and t.animacy=="inanimate" and p.text==("for" if state=="benefit" else "to") and p.relation==state and r.number=="plural" and r.animacy=="animate" and conn.text==("although" if state=="benefit" else "while")
def controls():
 ts=["The guards give the letters for the children although they send books for them.","The poets offer the seals for the children although they bring books for them.","The sailors show the letters for the children although they give books for them.","The captains send the seals for the children although they offer books for them.","The guards give the letters to the children while they send books to them.","The poets bring the seals to the children while they show books to them.","The sailors offer the letters to the children while they give books to them.","The captains send the seals to the children while they bring books to them.","The guards show some books for the children although they give the seals for them.","The poets give the letters for the children although they offer the books for them.","The sailors bring the seals to the children while they send the letters to them.","The captains offer books to the children while they show the seals to them.","The guards send the letters for the children although they bring the books for them.","The poets show the seals to the children while they give the letters to them.","The sailors give books for the children although they offer the seals for them.","The captains bring the letters to the children while they send the books to them.","The guards offer the seals for the children although they show the books for them.","The poets send the books to the children while they give the seals to them.","The sailors show the letters for the children although they bring the books for them.","The captains give the seals to the children while they offer the letters to them."]
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
    equations+=1;surface=li+ri;z=" ".join(x.text for x in surface)+".";a=audit(z);row={"rendered":z,"audit":a,"relation":state,"subject_antecedent":ls[0].ref,"subject_anaphor":rs[0].text,"recipient_antecedent":ls[4].ref,"recipient_anaphor":rs[4].text,"equations":eq,"provenance":{"construction":"joint plural subject/recipient anaphora equations","roles":[x.role for x in surface],"referents":[x.ref for x in surface],"numbers":[x.number for x in surface],"anaphoric_flags":[x.anaphoric for x in surface],"relations":[x.relation for x in surface],"independent_phrase_boundaries":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False},"reader_status":"unreviewed; exactness does not certify readability"};rows.append(row)
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
 survivors.sort(key=lambda x:x["audit"]["letters"],reverse=True);result={"experiment":EXPERIMENT_ID,"method":"joint plural subject/recipient anaphora equations","complete_prose_controls":controls(),"candidates":survivors,"equation_rows":rows[:200],"stats":{"relations":2,"states":states,"pruned":pruned,"feature_pruned":feature_pruned,"advances":advances,"equation_completions":equations,"exact":len(survivors)},"provenance":{"novelty_signature":SIGNATURE,"novelty_preflight":"fresh joint-plural signature; both pronoun links and plural agreement are closure constraints","joint_anaphoric_subject_recipient":True,"plural_agreement_live":True,"relation_state_retained":True,"recipient_theme_gate":True,"independent_pointer_sha_audit":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False,"next_construction":"add mixed singular/plural antecedent alternatives with explicit agreement branching","reader_next_test":"blind all 20 controls against word-shuffled controls before promoting any exact row"}}
 OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+"\n");return result
if __name__=="__main__":print(json.dumps(run(),indent=2))
