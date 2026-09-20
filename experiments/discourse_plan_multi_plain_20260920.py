"""Multi-plan plain discourse delayed realization with typed attributes."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
@dataclass(frozen=True)
class Subject: text:str; agreement:str
@dataclass(frozen=True)
class Plan:
 name:str; speakers:tuple[Subject,...]; e1:str; o1:tuple[str,...]; e2:str; valency:str; o2:tuple[str,...]; recipients:tuple[str,...]; referents:tuple[str,...]
S=lambda a,b: Subject(a,b)
PLANS=(
 Plan("weather",(S("I","first"),S("we","plural"),S("the forecaster","third")),"notice",("the storm","the rain","the dew"),"track","transitive",("the storm","the rain","the dew"),("the storm",),("storm","rain")),
 Plan("school",(S("I","first"),S("we","plural"),S("the teacher","third")),"help",("the student","the class","the child"),"guide","transitive",("the student","the class","the child"),("the student",),("student","class")),
 Plan("travel",(S("I","first"),S("we","plural"),S("the traveler","third")),"check",("the map","the route","the ticket"),"wait","intransitive",(),("the station",),("map","station")),
 Plan("meeting",(S("I","first"),S("we","plural"),S("the manager","third")),"review",("the plan","the notes","the report"),"send","ditransitive",("the update","a message","the report"),("the team","the staff","the client"),("plan","update","team")),
 Plan("letter",(S("I","first"),S("we","plural"),S("the writer","third")),"read",("the letter","the note","the message"),"send","ditransitive",("the reply","a note","the update"),("my friend","our friend","the editor"),("letter","reply","friend")),
 Plan("garden",(S("I","first"),S("we","plural"),S("the gardener","third")),"check",("the soil","the bed","the plants"),"water","transitive",("the plants","the garden","the bed"),("the plants",),("soil","plants")),
 Plan("work",(S("I","first"),S("we","plural"),S("the team","third")),"finish",("the task","the report","the project"),"share","transitive",("the results","the report","the plan"),("the client","the team","the manager"),("task","results","team")),
 Plan("home",(S("I","first"),S("we","plural"),S("the parent","third")),"check",("the child","the room","the door"),"give","ditransitive",("the meal","the book","the key"),("the child","the guest","the neighbor"),("child","meal","home")),
)
TENSES=(("present","simple"),("past","simple"),("present","progressive"),("past","progressive")); CONJ=("and","but","so")
IRREG={"send":"sent","give":"gave","read":"read"}; ING={"send":"sending","give":"giving","read":"reading"}
def letters(s): return re.sub(r"[^a-z]","",s.casefold())
def audit(s):
 t=letters(s); r=t[::-1]
 return {"letters":len(t),"two_pointer_exact":bool(t) and all(a==b for a,b in zip(t,r)),"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def consume(a,b):
 n=min(len(a),len(b))
 if a[:n]!=b[:n]: return None
 return a[n:],b[n:]
def surface(root,agreement,tense,aspect):
 if aspect=="simple":
  if tense=="present": return root+("s" if agreement=="third" else "")
  return IRREG.get(root,root+"d" if root.endswith("e") else root+"ed")
 aux=("is" if agreement=="third" else "am" if agreement=="first" else "are") if tense=="present" else ("were" if agreement=="plural" else "was")
 progressive=ING.get(root,(root[:-1] if root.endswith("e") else root)+"ing")
 return aux+" "+progressive
def grammatical(text):
 ws=re.findall(r"[a-z]+",text.casefold())
 if any((w=="a" and i+1<len(ws) and ws[i+1][0] in "aeiou") or (w=="an" and i+1<len(ws) and ws[i+1][0] not in "aeiou") for i,w in enumerate(ws)): return False
 return not re.search(r"^(and|but|so)\b|\b(and|but|so)$",text.casefold())
def strict_surface(text):
 if not grammatical(text): return False
 low=text.casefold()
 if re.search(r"noticeed|noticeing|trackk|waitted|sended|\bI is\b|\bI are\b|\bwe is\b|\bwe am\b|\bthe [a-z]+ am\b|\bthe [a-z]+ are\b",low): return False
 return True
def render(xs): return " ".join(xs).replace(" but ",", but ").replace(" so ",", so ")
def run(limit=50000):
 states=pruned=0; controls=[]; exact=[]; seen=set()
 for p in PLANS:
  for ti,(tense,aspect) in enumerate(TENSES):
   subj=p.speakers[ti%3]; v1=surface(p.e1,subj.agreement,tense,aspect); v2=surface(p.e2,subj.agreement,tense,aspect); tail=[] if p.valency=="intransitive" else [p.o2[ti%len(p.o2)]] if p.valency=="transitive" else [p.recipients[ti%len(p.recipients)],p.o2[ti%len(p.o2)]]
   text=render([subj.text,v1,p.o1[ti%len(p.o1)],CONJ[ti%3],v2]+tail)
   if strict_surface(text): controls.append({"rendered":text,"audit":audit(text),"plan":p.name,"tense":tense,"aspect":aspect,"valency":p.valency,"agreement":subj.agreement,"referents":p.referents,"complete_utterance":True})
 for p in PLANS:
  for tense,aspect in TENSES:
   for subj in p.speakers:
    roles=("SUBJ","V1","O1","CONJ","V2")+("O2",) if p.valency=="transitive" else (("SUBJ","V1","O1","CONJ","V2") if p.valency=="intransitive" else ("SUBJ","V1","O1","CONJ","V2","RECIP","O2"))
    domains={"SUBJ":(subj.text,),"V1":(surface(p.e1,subj.agreement,tense,aspect),),"O1":p.o1,"CONJ":CONJ,"V2":(surface(p.e2,subj.agreement,tense,aspect),),"O2":p.o2,"RECIP":p.recipients}; rr=tuple(reversed(roles)); stack=[]
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
       seen.add(text); exact.append({"rendered":text,"audit":a,"provenance":{"plan":p.name,"tense":tense,"aspect":aspect,"valency":p.valency,"agreement":subj.agreement,"referent_binding":p.referents,"delayed_surface_realization":True,"finished_tape_reversal":False,"posthoc_repair":False,"mirrored_token_units":False,"catalogue_replay":False,"word_path":prov}})
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
 return {"method":"discourse-plan-multi-plain-20260920","plans":len(PLANS),"tense_aspect_states":TENSES,"states":states,"pruned":pruned,"max_letters":80,"controls":controls,"control_count":len(controls),"exact_candidates":exact,"candidate_count":len(exact),"status":"reader gate required" if exact else "representation bottleneck","provenance":{"signature":"discourse-plan-conditioned|delayed-surface-realization|attribute-pushdown|joint-character-output|typed-valency|agreement|referent-binding|tense-aspect|multi-plan","ordinary_contemporary_plans":True,"independent_pointer_sha":True,"novelty_preflight":"eight-plan expansion of corrected plain discourse grammar"},"next_construction":"add aspectual adverb attributes plan-by-plan without widening lexical banks"}
if __name__=="__main__":
 d=run(); (ROOT/"runs/discourse-plan-multi-plain-20260920.json").write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d,indent=2))
