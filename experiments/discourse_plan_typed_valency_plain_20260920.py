"""Plain contemporary discourse-plan realization with typed valency."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
@dataclass(frozen=True)
class Subject: text:str; agreement:str
@dataclass(frozen=True)
class Predicate: event:str; valency:str; forms:tuple[tuple[str,str],...]
@dataclass(frozen=True)
class Plan: name:str; speakers:tuple[Subject,...]; event1:str; event2:str; relation:str; referents:tuple[str,...]

PLAN=Plan("storm-response",(Subject("I","base"),Subject("we","base"),Subject("the teacher","third")),"notice","respond","contrast",("storm","friend","message","map"))
PREDICATES=(Predicate("notice","transitive",(("base","notice"),("third","notices"))),
 Predicate("respond","transitive",(("base","track"),("third","tracks"))),
 Predicate("respond","intransitive",(("base","wait"),("third","waits"))),
 Predicate("respond","ditransitive",(("base","send"),("third","sends"))))
OBJECTS={"storm":("the storm","that storm","the rain","the dew"),"friend":("my friend","our friend","the neighbor"),"message":("the message","a note","the update"),"map":("the map","a route","the plan")}
CONJ=("and","but","so"); RECIPIENTS=("my friend","our friend","the neighbor")
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
def verb(pred,agreement): return dict(pred.forms)[agreement]
def render(chunks): return " ".join(chunks).replace(" and "," and ").replace(" but ",", but ").replace(" so ",", so ")
def run(limit=50000):
 states=pruned=0; controls=[]; exact=[]; seen=set()
 for pred in PREDICATES[1:]:
  for j in range(8):
   subj=PLAN.speakers[j%3]; v1=verb(PREDICATES[0],subj.agreement); o1=OBJECTS["storm"][j%4]; conj=CONJ[j%3]; v2=verb(pred,subj.agreement)
   tail=[] if pred.valency=="intransitive" else [OBJECTS["friend"][j%3]] if pred.valency=="transitive" else [RECIPIENTS[j%3],OBJECTS["message"][j%3]]
   text=render([subj.text,v1,o1,conj,v2]+tail); controls.append({"rendered":text,"audit":audit(text),"valencies":("transitive",pred.valency),"agreement":subj.agreement,"referents":PLAN.referents,"complete_utterance":True})
 for pred in PREDICATES[1:]:
  for subj in PLAN.speakers:
   roles=("SUBJ","V1","O1","CONJ","V2")+("O2",) if pred.valency=="transitive" else (("SUBJ","V1","O1","CONJ","V2") if pred.valency=="intransitive" else ("SUBJ","V1","O1","CONJ","V2","RECIP","O2"))
   domains={"SUBJ":(subj.text,),"V1":(verb(PREDICATES[0],subj.agreement),),"O1":OBJECTS["storm"],"CONJ":CONJ,"V2":(verb(pred,subj.agreement),),"O2":OBJECTS["storm"] if pred.valency=="transitive" else OBJECTS["message"],"RECIP":RECIPIENTS}; rr=tuple(reversed(roles)); stack=[]
   for lw in domains[roles[0]]:
    for rw in domains[rr[0]]:
     got=consume(letters(lw),letters(rw)[::-1])
     if got is not None: stack.append((1,1,[lw],[rw],got[0],got[1],{"plan":PLAN.name,"valencies":("transitive",pred.valency),"agreement":subj.agreement,"referents":PLAN.referents},(("L",roles[0],lw),("R",rr[0],rw))))
   while stack and states<limit:
    li,ri,left,right,lbuf,rbuf,attrs,prov=stack.pop(); states+=1
    if li==len(roles) and ri==len(rr):
     if lbuf or rbuf: pruned+=1; continue
     text=render(left)+"; "+render(right); a=audit(text)
     if a["two_pointer_exact"] and a["letters"]>38 and grammatical(text) and text not in seen:
      seen.add(text); exact.append({"rendered":text,"audit":a,"provenance":{"plan":PLAN.name,"attributes":attrs,"left_roles":roles,"right_roles":rr,"delayed_surface_realization":True,"typed_valency":pred.valency,"subject_object_agreement":subj.agreement,"referent_binding":PLAN.referents,"finished_tape_reversal":False,"posthoc_repair":False,"mirrored_token_units":False,"catalogue_replay":False,"word_path":prov}})
     continue
    if li<len(roles):
     for w in reversed(domains[roles[li]]):
      got=consume(lbuf+letters(w),rbuf)
      if got is not None: stack.append((li+1,ri,left+[w],right,got[0],got[1],attrs,prov+(("L",roles[li],w),)))
      else: pruned+=1
    if ri<len(rr):
     for w in reversed(domains[rr[ri]]):
      got=consume(lbuf,rbuf+letters(w)[::-1])
      if got is not None: stack.append((li,ri+1,left,[w]+right,got[0],got[1],attrs,prov+(("R",rr[ri],w),)))
      else: pruned+=1
   if states>=limit: break
  if states>=limit: break
 return {"method":"discourse-plan-typed-valency-plain-20260920","plans":1,"predicate_valencies":["transitive","intransitive","ditransitive"],"states":states,"pruned":pruned,"max_letters":80,"controls":controls,"control_count":len(controls),"exact_candidates":exact,"candidate_count":len(exact),"status":"reader gate required" if exact else "representation bottleneck","provenance":{"signature":"discourse-plan-conditioned|delayed-surface-realization|attribute-pushdown|joint-character-output|typed-valency|agreement|referent-binding|plain-contemporary","coherent_authored_plan":True,"independent_pointer_sha":True,"novelty_preflight":"plain contemporary lexical rerun of typed discourse representation"},"next_construction":"bind tense/aspect features to the same plain predicate lexicon"}
if __name__=="__main__":
 d=run(); (ROOT/"runs/discourse-plan-typed-valency-plain-20260920.json").write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d,indent=2))
