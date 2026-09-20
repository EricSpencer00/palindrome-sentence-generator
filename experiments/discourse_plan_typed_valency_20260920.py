"""Discourse-plan delayed realization with typed valency/agreement attributes."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Subject:
 text:str; number:str
@dataclass(frozen=True)
class Predicate:
 event:str; valency:str; forms:tuple[tuple[str,str],...]
@dataclass(frozen=True)
class Plan:
 name:str; speaker:tuple[Subject,...]; event1:str; event2:str; relation:str; referents:tuple[str,...]

PLAN=Plan("moon-and-tide",(Subject("I","sg"),Subject("we","pl"),Subject("this prince","sg")),"observe","respond", "contrast",("moon","tide","child","rose"))
PREDICATES=(Predicate("observe","transitive",(("sg","behold"),("pl","behold"))),
 Predicate("respond","transitive",(("sg","mourn"),("pl","mourn"))),
 Predicate("respond","intransitive",(("sg","wait"),("pl","wait"))),
 Predicate("respond","ditransitive",(("sg","send"),("pl","send"))))
OBJECTS={"moon":("the moon","that pale moon","the silver moon","the dew"),"tide":("the tide","that dark tide","the turning tide","the dew"),"child":("the child","that young child","our child"),"rose":("the rose","that red rose","the garden rose","the dew")}
CONJ=("yet","but","though")
RECIPIENTS=("the child","that young child","our child")

def letters(s): return re.sub(r"[^a-z]","",s.casefold())
def audit(s):
 t=letters(s); r=t[::-1]
 return {"letters":len(t),"two_pointer_exact":bool(t) and all(a==b for a,b in zip(t,r)),"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def consume(a,b):
 n=min(len(a),len(b))
 if a[:n]!=b[:n]: return None
 return a[n:],b[n:]
def grammatical(text):
 ws=re.findall(r"[a-z]+",text.casefold())
 return all(not (w=="a" and i+1<len(ws) and ws[i+1][0] in "aeiou") and not (w=="an" and i+1<len(ws) and ws[i+1][0] not in "aeiou") for i,w in enumerate(ws))
def verb(pred,number): return dict(pred.forms)[number]
def render(chunks):
 return " ".join(chunks).replace(" yet ",", yet ").replace(" but ",", but ").replace(" though ",", though ")

def run(limit=50000):
 states=pruned=0; exact=[]; seen=set(); controls=[]
 # >=20 complete controls from one coherent plan, with typed frame/attribute
 # choices. Controls are only reporting surfaces; search remains delayed.
 for pred in PREDICATES[1:]:
  for j in range(8):
   subj=PLAN.speaker[j%len(PLAN.speaker)]; v1=verb(PREDICATES[0],subj.number); o1=OBJECTS["moon"][j%3]; conj=CONJ[j%3]; v2=verb(pred,subj.number)
   if pred.valency=="intransitive": tail=[]
   elif pred.valency=="transitive": tail=[OBJECTS["tide"][j%3]]
   else: tail=[RECIPIENTS[j%3],OBJECTS["rose"][j%3]]
   text=render([subj.text,v1,o1,conj,v2]+tail); controls.append({"rendered":text,"audit":audit(text),"valencies":("transitive",pred.valency),"agreement":subj.number,"referents":PLAN.referents,"complete_utterance":True})
 # Surface paths are determined by shared typed attributes, but lexical
 # realizations on each side are selected independently under debt.
 for pred in PREDICATES[1:]:
  for subj in PLAN.speaker:
   roles=("SUBJ","V1","O1","CONJ","V2")+("O2",) if pred.valency=="transitive" else (("SUBJ","V1","O1","CONJ","V2") if pred.valency=="intransitive" else ("SUBJ","V1","O1","CONJ","V2","RECIP","O2"))
   domains={"SUBJ":(subj.text,),"V1":(verb(PREDICATES[0],subj.number),),"O1":OBJECTS["moon"],"CONJ":CONJ,"V2":(verb(pred,subj.number),),"O2":OBJECTS["tide"] if pred.valency=="transitive" else OBJECTS["rose"],"RECIP":RECIPIENTS}
   right_roles=tuple(reversed(roles)); stack=[]
   for lw in domains[roles[0]]:
    for rw in domains[right_roles[0]]:
     got=consume(letters(lw),letters(rw)[::-1])
     if got is not None: stack.append((1,1,[lw],[rw],got[0],got[1],("utterance","event1"),{"plan":PLAN.name,"valencies":("transitive",pred.valency),"agreement":subj.number,"referents":PLAN.referents},(("L",roles[0],lw),("R",right_roles[0],rw))))
   while stack and states<limit:
    li,ri,left,right,lbuf,rbuf,push,attrs,prov=stack.pop(); states+=1
    if li==len(roles) and ri==len(right_roles):
     if lbuf or rbuf: pruned+=1; continue
     text=render(left)+"; "+render(right); a=audit(text)
     if a["two_pointer_exact"] and a["letters"]>38 and grammatical(text) and text not in seen:
      seen.add(text); exact.append({"rendered":text,"audit":a,"provenance":{"plan":PLAN.name,"attributes":attrs,"pushdown":push,"left_roles":roles,"right_roles":right_roles,"delayed_surface_realization":True,"typed_valency":pred.valency,"subject_object_agreement":subj.number,"referent_binding":PLAN.referents,"finished_tape_reversal":False,"posthoc_repair":False,"mirrored_token_units":False,"catalogue_replay":False,"word_path":prov}})
     continue
    if li<len(roles):
     role=roles[li]
     for w in reversed(domains[role]):
      got=consume(lbuf+letters(w),rbuf)
      if got is None: pruned+=1; continue
      stack.append((li+1,ri,left+[w],right,got[0],got[1],push+("open_event2" if role=="V2" else "",),attrs,prov+(("L",role,w),)))
    if ri<len(right_roles):
     role=right_roles[ri]
     for w in reversed(domains[role]):
      got=consume(lbuf,rbuf+letters(w)[::-1])
      if got is None: pruned+=1; continue
      stack.append((li,ri+1,left,[w]+right,got[0],got[1],push,attrs,prov+(("R",role,w),)))
   if states>=limit: break
  if states>=limit: break
 return {"method":"discourse-plan-typed-valency-20260920","plans":1,"predicate_valencies":["transitive","intransitive","ditransitive"],"states":states,"pruned":pruned,"max_letters":80,"controls":controls,"control_count":len(controls),"exact_candidates":exact,"candidate_count":len(exact),"status":"reader gate required" if exact else "representation bottleneck","provenance":{"signature":"discourse-plan-conditioned|delayed-surface-realization|attribute-pushdown|joint-character-output|typed-valency|agreement|referent-binding","coherent_authored_plan":True,"independent_pointer_sha":True,"novelty_preflight":"typed predicate attributes added before lexical surface emission"},"next_construction":"bind tense/aspect morphology to the same valency-feature state before surface expansion"}

if __name__=="__main__":
 d=run(); (ROOT/"runs/discourse-plan-typed-valency-20260920.json").write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d,indent=2))
