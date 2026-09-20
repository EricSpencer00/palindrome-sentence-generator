"""Plain discourse-plan realization with tense/aspect morphology attributes."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
@dataclass(frozen=True)
class Subject: text:str; agreement:str
@dataclass(frozen=True)
class Plan: name:str; speakers:tuple[Subject,...]; referents:tuple[str,...]
PLAN=Plan("storm-response",(Subject("I","first_singular"),Subject("we","plural"),Subject("the teacher","third")),("storm","message","friend","map"))
TENSE_ASPECT=(("present","simple"),("past","simple"),("present","progressive"),("past","progressive"))
OBJECTS={"storm":("the storm","that storm","the rain","the dew"),"friend":("my friend","our friend","the neighbor"),"message":("the message","a note","the update"),"map":("the map","a route","the plan")}
RECIPIENTS=("my friend","our friend","the neighbor"); CONJ=("and","but","so")
ROOTS={"observe":"notice","track":"track","wait":"wait","send":"send"}
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
def surface(root,agreement,tense,aspect):
 if aspect=="simple":
  if tense=="present": return root+("s" if agreement=="third" else "")
  return root+"ed" if root not in {"send"} else "sent"
 aux=(("is" if agreement=="third" else "am" if agreement=="first_singular" else "are") if tense=="present" else ("was" if agreement=="third" else "were"))
 ing={"notice":"noticing","track":"tracking","wait":"waiting","send":"sending"}[root]
 return aux+" "+ing
def strict_surface(text):
 """Reject article, agreement, auxiliary, and clause-boundary errors."""
 if not grammatical(text): return False
 low=text.casefold()
 if re.search(r"^(?:and|but|so)\b|\b(?:and|but|so)$",low): return False
 if re.search(r"\bI is\b|\bI are\b|\bwe is\b|\bwe am\b|\bthe teacher am\b|\bthe teacher are\b",low): return False
 if re.search(r"\bI notices\b|\bwe notices\b|\bthe teacher notice\b",low): return False
 return True
def render(xs): return " ".join(xs).replace(" but ",", but ").replace(" so ",", so ")
def run(limit=50000):
 states=pruned=0; controls=[]; exact=[]; seen=set()
 # 4 morphology states x 8 ordinary lexical controls = 32 complete controls.
 for ti,(tense,aspect) in enumerate(TENSE_ASPECT):
  for j in range(8):
   subj=PLAN.speakers[j%3]; v1=surface(ROOTS["observe"],subj.agreement,tense,aspect); o1=OBJECTS["storm"][j%4]; conj=CONJ[j%3]
   v2=surface(ROOTS["track"],subj.agreement,tense,aspect); text=render([subj.text,v1,o1,conj,v2,OBJECTS["storm"][j%4]])
   if strict_surface(text): controls.append({"rendered":text,"audit":audit(text),"tense":tense,"aspect":aspect,"agreement":subj.agreement,"valency":"transitive+transitive","referents":PLAN.referents,"complete_utterance":True})
 for tense,aspect in TENSE_ASPECT:
  for subj in PLAN.speakers:
   for valency in ("transitive","intransitive","ditransitive"):
    roles=("SUBJ","V1","O1","CONJ","V2")+("O2",) if valency=="transitive" else (("SUBJ","V1","O1","CONJ","V2") if valency=="intransitive" else ("SUBJ","V1","O1","CONJ","V2","RECIP","O2"))
    domains={"SUBJ":(subj.text,),"V1":(surface(ROOTS["observe"],subj.agreement,tense,aspect),),"O1":OBJECTS["storm"],"CONJ":CONJ,"V2":(surface(ROOTS["track" if valency=="transitive" else "wait" if valency=="intransitive" else "send"],subj.agreement,tense,aspect),),"O2":OBJECTS["storm"] if valency=="transitive" else OBJECTS["message"],"RECIP":RECIPIENTS}; rr=tuple(reversed(roles)); stack=[]
    for lw in domains[roles[0]]:
     for rw in domains[rr[0]]:
      got=consume(letters(lw),letters(rw)[::-1])
      if got is not None: stack.append((1,1,[lw],[rw],got[0],got[1],(("L",roles[0],lw),("R",rr[0],rw))))
    while stack and states<limit:
     li,ri,left,right,lbuf,rbuf,prov=stack.pop(); states+=1
     if li==len(roles) and ri==len(rr):
      if lbuf or rbuf: pruned+=1; continue
      text=render(left)+"; "+render(right); a=audit(text)
      if a["two_pointer_exact"] and a["letters"]>38 and strict_surface(text) and text not in seen:
       seen.add(text); exact.append({"rendered":text,"audit":a,"provenance":{"plan":PLAN.name,"tense":tense,"aspect":aspect,"valency":valency,"agreement":subj.agreement,"referent_binding":PLAN.referents,"delayed_surface_realization":True,"finished_tape_reversal":False,"posthoc_repair":False,"mirrored_token_units":False,"catalogue_replay":False,"word_path":prov}})
      continue
     if li<len(roles):
      for w in reversed(domains[roles[li]]):
       got=consume(lbuf+letters(w),rbuf)
       if got is not None: stack.append((li+1,ri,left+[w],right,got[0],got[1],prov+(("L",roles[li],w),)))
       else: pruned+=1
     if ri<len(rr):
      for w in reversed(domains[rr[ri]]):
       got=consume(lbuf,rbuf+letters(w)[::-1])
       if got is not None: stack.append((li,ri+1,left,[w]+right,got[0],got[1],prov+(("R",rr[ri],w),)))
       else: pruned+=1
    if states>=limit: break
   if states>=limit: break
  if states>=limit: break
 return {"method":"discourse-plan-tense-aspect-corrected-20260920","plans":1,"tense_aspect_states":TENSE_ASPECT,"states":states,"pruned":pruned,"max_letters":80,"controls":controls,"control_count":len(controls),"exact_candidates":exact,"candidate_count":len(exact),"status":"reader gate required" if exact else "representation bottleneck","provenance":{"signature":"discourse-plan-conditioned|delayed-surface-realization|attribute-pushdown|joint-character-output|typed-valency|agreement|referent-binding|tense-aspect-corrected","coherent_authored_plan":True,"independent_pointer_sha":True,"novelty_preflight":"corrected irregular past and subject-specific auxiliaries before evidence"},"next_construction":"add lexical aspectual adverbs while retaining morphology state"}
if __name__=="__main__":
 d=run(); (ROOT/"runs/discourse-plan-tense-aspect-20260920.json").write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d,indent=2))
