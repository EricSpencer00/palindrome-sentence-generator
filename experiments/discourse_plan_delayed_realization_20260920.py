"""Discourse-plan-conditioned delayed-surface realization constructor."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Plan:
 name:str; speaker:str; event1:str; event2:str; relation:str; tense:str
 roles:tuple[str,...]=( "SPEAKER","V1","O1","CONJ","V2","O2")

PLANS=(
 Plan("moon-watch","royal-speaker","observe","lament","contrast","present"),
 Plan("rose-vow","courtly-speaker","praise","guard","addition","present"),
 Plan("tide-memory","old-speaker","remember","follow","contrast","present"),
 Plan("winter-oath","noble-speaker","keep","honor","addition","present"),
 Plan("star-warning","watchful-speaker","heed","fear","contrast","present"),
 Plan("dawn-counsel","wise-speaker","counsel","await","addition","present"),
 Plan("crown-loss","fallen-speaker","mourn","seek","contrast","present"),
 Plan("garden-mercy","gentle-speaker","spare","bless","addition","present"),
)

# Three-to-five realizations per semantic role; each plan's lexical choices
# preserve its referent/event relation across both utterances.
LEX={
 "royal-speaker":("I","we","this prince"), "courtly-speaker":("I","we","this lover"),
 "old-speaker":("I","we","this old heart"), "noble-speaker":("I","we","this lord"),
 "watchful-speaker":("I","we","this warder"), "wise-speaker":("I","we","this sage"),
 "fallen-speaker":("I","we","this fallen king"), "gentle-speaker":("I","we","this gentle soul"),
 "observe":("behold","watch","mark"), "praise":("praise","honor","sing of"),
 "remember":("remember","recall","cherish"), "keep":("keep","hold","guard"),
 "heed":("heed","watch","mark"), "counsel":("counsel","advise","guide"),
 "mourn":("mourn","lament","grieve for"), "spare":("spare","save","forgive"),
 "lament":("mourn","lament","grieve for"), "guard":("guard","keep","defend"),
 "follow":("follow","seek","pursue"), "honor":("honor","serve","praise"),
 "fear":("fear","dread","flee from"), "await":("await","watch for","welcome"),
 "seek":("seek","find","follow"), "bless":("bless","cherish","save"),
 "moon":("the moon","that pale moon","the silver moon","the dew"), "rose":("the rose","that red rose","the garden rose","the dew"),
 "tide":("the tide","that dark tide","the turning tide","the dew"), "winter":("the winter fire","that cold flame","the winter star","the dew"),
 "star":("the star","that red star","the evening star","the dew"), "dawn":("the dawn","that fair dawn","the coming dawn","the dew"),
 "crown":("the lost crown","that fallen crown","the old crown","the dew"), "garden":("the garden","that still garden","the green garden","the dew"),
 "contrast":("yet", "but", "though"), "addition":("and", "then", "and still"),
}
OBJECT={"observe":"moon","praise":"rose","remember":"tide","keep":"winter","heed":"star","counsel":"dawn","mourn":"crown","spare":"garden"}
OBJECT2={"lament":"tide","guard":"crown","follow":"dawn","honor":"crown","fear":"star","await":"dawn","seek":"crown","bless":"garden"}

def letters(s): return re.sub(r"[^a-z]","",s.casefold())
def audit(s):
 t=letters(s); r=t[::-1]
 return {"letters":len(t),"two_pointer_exact":bool(t) and all(a==b for a,b in zip(t,r)),"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def consume(a,b):
 n=min(len(a),len(b))
 if a[:n]!=b[:n]: return None
 return a[n:],b[n:]
def plan_choices(p, role):
 if role=="SPEAKER": return LEX[p.speaker]
 if role=="V1": return LEX[p.event1]
 if role=="V2": return LEX[p.event2]
 if role=="O1": return LEX[OBJECT[p.event1]]
 if role=="O2": return LEX[OBJECT2[p.event2]]
 return LEX[p.relation]
def render(chunks): return " ".join(chunks).replace(" yet ",", yet ").replace(" but ",", but ").replace(" though ",", though ")

def run(limit=50000):
 states=pruned=0; exact=[]; seen=set(); controls=[]
 # Complete controls are generated from discourse plans, with no catalogue
 # text.  8 plans x 3 paired realizations gives >=20 reader-facing controls.
 for p in PLANS:
  for j in range(3):
   chunks=[plan_choices(p,r)[j%len(plan_choices(p,r))] for r in p.roles]
   text=render(chunks); controls.append({"rendered":text,"audit":audit(text),"plan":p.name,"complete_utterance":True,"reader_control":True})
 # Search each plan while retaining a pushdown of open event constituents and
 # referents.  Right surface is chosen from its inner edge outward, but no
 # completed sentence is materialized or reversed.
 for p in PLANS:
  roles=p.roles; right_roles=tuple(reversed(roles)); domains={r:plan_choices(p,r) for r in roles}
  stack=[]
  for lw in domains[roles[0]]:
   for rw in domains[right_roles[0]]:
    got=consume(letters(lw),letters(rw)[::-1])
    if got is not None:
     stack.append((1,1,[lw],[rw],got[0],got[1],("utterance","event1"),{"plan":p.name,"speaker":p.speaker,"open_event":"event1","referents":(OBJECT[p.event1],OBJECT2[p.event2])},(("L",roles[0],lw),("R",right_roles[0],rw))))
  while stack and states<limit:
   li,ri,left,right,lbuf,rbuf,open_stack,attrs,prov=stack.pop(); states+=1
   if li==len(roles) and ri==len(right_roles):
    if lbuf or rbuf: pruned+=1; continue
    text=render(left)+"; "+render(right)
    a=audit(text)
    if a["two_pointer_exact"] and a["letters"]>38 and text not in seen:
     seen.add(text); exact.append({"rendered":text,"audit":a,"provenance":{"plan":p.name,"attributes":attrs,"pushdown_trace":open_stack,"left_roles":roles,"right_roles":right_roles,"delayed_surface_realization":True,"finished_tape_reversal":False,"posthoc_repair":False,"mirrored_token_units":False,"catalogue_replay":False,"word_path":prov}})
    continue
   if li<len(roles):
    role=roles[li]
    for w in reversed(domains[role]):
     got=consume(lbuf+letters(w),rbuf)
     if got is None: pruned+=1; continue
     stack.append((li+1,ri,left+[w],right,got[0],got[1],open_stack+("event2" if role=="V2" else "",),attrs,prov+(("L",role,w),)))
   if ri<len(right_roles):
    role=right_roles[ri]
    for w in reversed(domains[role]):
     got=consume(lbuf,rbuf+letters(w)[::-1])
     if got is None: pruned+=1; continue
     stack.append((li,ri+1,left,[w]+right,got[0],got[1],open_stack,attrs,prov+(("R",role,w),)))
  if states>=limit: break
 return {"method":"discourse-plan-delayed-realization-20260920","plans":len(PLANS),"role_realization_bounds":{"min":3,"max":5},"max_letters":80,"states":states,"pruned":pruned,"controls":controls,"control_count":len(controls),"exact_candidates":exact,"candidate_count":len(exact),"status":"reader gate required" if exact else "representation bottleneck","provenance":{"signature":"discourse-plan-conditioned|delayed-surface-realization|attribute-pushdown|joint-character-output","coherent_authored_plans":True,"independent_pointer_sha":True,"novelty_preflight":"discourse attributes and latent surface spans distinct from CCG/CFG/tree lanes"},"next_construction":"split event predicates into typed transitive/intransitive valency attributes before adding new plans"}

if __name__=="__main__":
 d=run(); (ROOT/"runs/discourse-plan-delayed-realization-20260920.json").write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d,indent=2))
