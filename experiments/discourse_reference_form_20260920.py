"""Two-clause discourse with jointly selected definite/pronominal reference."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
@dataclass(frozen=True)
class Scene:
 name:str; subject:str; v1:tuple[str,...]; referent:tuple[str,...]; agreement:str; pronouns:tuple[str,...]; v2:tuple[str,...]; object2:tuple[str,...]
SCENES=(
 Scene("classroom","the teacher",("helps","guides","greets"),("the student","the child","the class"),"singular",("they","the student","the child"),("guides","checks","encourages"),("the class","the work","the team")),
 Scene("meeting","the manager",("sends","shares","reviews"),("the message","the update","the report"),"singular",("it","the message","the update"),("reaches","changes","helps"),("the team","the plan","the staff")),
 Scene("garden","the gardener",("waters","checks","plants"),("the plant","the tree","the garden"),"singular",("it","the plant","the tree"),("grows","needs","changes"),("the sun","the soil","the yard")),
 Scene("travel","the traveler",("checks","carries","packs"),("the ticket","the map","the bag"),"singular",("it","the ticket","the map"),("opens","shows","holds"),("the gate","the route","the plan")),
 Scene("home","the parent",("calls","helps","watches"),("the child","the guest","the dog"),"singular",("they","the child","the guest"),("answers","finishes","joins"),("the meal","the game","the family")),
)
RELATIONS=("and","but","so","because")
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
 if any((w=="a" and i+1<len(ws) and ws[i+1][0] in "aeiou") or (w=="an" and i+1<len(ws) and ws[i+1][0] not in "aeiou") for i,w in enumerate(ws)): return False
 return not re.search(r"^(and|but|so|because)\b|\b(and|but|so|because)$",text.casefold())
def render(xs): return " ".join(xs).replace(" and "," and ").replace(" but ",", but ").replace(" so ",", so ").replace(" because "," because ")
def agree_verb(word, reference):
 if reference!="they": return word
 return {"guides":"guide","checks":"check","encourages":"encourage","reaches":"reach","changes":"change","helps":"help","grows":"grow","needs":"need","opens":"open","shows":"show","holds":"hold","answers":"answer","finishes":"finish","joins":"join"}.get(word,word)
def run(limit=50000):
 states=pruned=0; controls=[]; exact=[]; seen=set()
 # Reference form is selected as an attribute of the bound referent before
 # either side enters the character-level stack.
 for s in SCENES:
  for j in range(5):
   ref=s.pronouns[j%len(s.pronouns)]; text=render([s.subject,s.v1[j%3],s.referent[j%3],RELATIONS[j%4],ref,agree_verb(s.v2[j%3],ref),s.object2[j%3]])
   if grammatical(text): controls.append({"rendered":text,"audit":audit(text),"scene":s.name,"relation":RELATIONS[j%4],"reference_form":ref,"referent":s.referent[j%3],"agreement":s.agreement,"complete_utterance":True})
 for s in SCENES:
  for relation in RELATIONS:
   for ref_i,ref in enumerate(s.pronouns):
    # Shared discourse attributes are fixed now; lexical spans remain latent.
    roles=("SUBJ","V1","O1","REL","REF","V2","O2"); domains={"SUBJ":(s.subject,),"V1":s.v1,"O1":s.referent,"REL":(relation,),"REF":(ref,),"V2":tuple(agree_verb(v,ref) for v in s.v2),"O2":s.object2}; rr=tuple(reversed(roles)); stack=[]
    for lw in domains[roles[0]]:
     for rw in domains[rr[0]]:
      got=consume(letters(lw),letters(rw)[::-1])
      if got is not None: stack.append((1,1,[lw],[rw],got[0],got[1],{"scene":s.name,"relation":relation,"reference_form":ref,"referent_agreement":s.agreement},(("L",roles[0],lw),("R",rr[0],rw))))
    while stack and states<limit:
     li,ri,left,right,lbuf,rbuf,attrs,prov=stack.pop(); states+=1
     if li==len(roles) and ri==len(rr):
      if lbuf or rbuf: pruned+=1; continue
      text=render(left)+"; "+render(right); a=audit(text)
      if a["two_pointer_exact"] and a["letters"]>38 and grammatical(text) and text not in seen:
       seen.add(text); exact.append({"rendered":text,"audit":a,"provenance":{"scene":s.name,"relation":relation,"reference_form":ref,"referent_agreement":s.agreement,"joint_reference_attribute":True,"delayed_surface_realization":True,"finished_tape_reversal":False,"posthoc_repair":False,"mirrored_token_units":False,"catalogue_replay":False,"word_path":prov}})
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
  if states>=limit: break
 return {"method":"discourse-reference-form-20260920","scenes":len(SCENES),"relations":len(RELATIONS),"states":states,"pruned":pruned,"controls":controls,"control_count":len(controls),"exact_candidates":exact,"candidate_count":len(exact),"status":"reader gate required" if exact else "representation bottleneck","provenance":{"signature":"discourse-plan-conditioned|reference-form-binding|delayed-surface-realization|joint-character-output","definite_or_pronoun_attribute":True,"independent_pointer_sha":True,"novelty_preflight":"reference-form choice bound to discourse referent before character output"},"next_construction":"add plural referents and agreement-sensitive pronoun domains while retaining bound reference forms"}
if __name__=="__main__":
 d=run(); (ROOT/"runs/discourse-reference-form-20260920.json").write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d,indent=2))
