"""Recursive semantic-frame grammar with live character obligations."""
from __future__ import annotations
import argparse,json
from hashlib import sha256
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import normalize_letters

ROLES=("baker","nurse","poet","teacher")
VERBS={"baker":"bakes","nurse":"helps","poet":"writes","teacher":"guides"}
OBJECT={"baker":"bread","nurse":"a note","poet":"a lesson","teacher":"the child"}
def derivations(depth=2):
 out=[]
 def rec(used,words,frames):
  if frames:
   out.append((tuple(words),tuple(frames)))
  if len(frames)>=depth:return
  for role in ROLES:
   if role in used:continue
   # A recursive frame is an ordinary coordinated discourse, not a mirrored unit.
   rec(used|{role},words+list(("the",role,VERBS[role])+tuple(OBJECT[role].split())),frames+[(role,"SVO")])
 rec(set(),[],[]);return out
def audit(text):
 n=normalize_letters(text);return {"exact":n==n[::-1] and all(n[i]==n[-1-i] for i in range(len(n)//2)),"letters":len(n),"sha256":sha256(n.encode()).hexdigest(),"two_pointer_pairs":len(n)//2}
def product(left,right,cap=4000):
 # Character automata over recursive derivation words; boundaries are epsilon.
 L=list(left);R=list(right); states={(0,0,len(R)-1,len(R[-1])-1)}; frontier=[]
 while states and len(frontier)<cap:
  lw,lc,rw,rc=states.pop();
  # The left cursor moves upward from 0, while the right cursor moves
  # downward from the final word.  A completed product therefore ends at
  # ``lw == len(L)`` and ``rw < 0``; comparing the descending cursor with
  # ``len(R)`` made the success state unreachable for every derivation.
  if lw==len(L) and rw<0:return True,frontier
  if lw<len(L) and lc==len(L[lw]):states.add((lw+1,0,rw,rc));continue
  if rw<len(R) and rc<0:states.add((lw,lc,rw-1,len(R[rw-1])-1));continue
  if lw>=len(L) or rw<0:continue
  if lc>=len(L[lw]) or rc<0:continue
  if L[lw][lc].lower()!=R[rw][rc].lower():
   frontier.append((lw,lc,rw,rc,L[lw][lc],R[rw][rc]));continue
  states.add((lw,lc+1,rw,rc-1))
 return False,frontier
def run():
 ps=derivations();closures=[];w=[];states=0
 for li,(lp,lf) in enumerate(ps):
  for ri,(rp,rf) in enumerate(ps):
   ok,front=product(lp,rp);states+=len(front)+1
   # Render both independently authored derivations in ordinary order; only
   # the product's right automaton runs backward for obligations.
   text=" ".join(lp+rp);a=audit(text)
   row={"text":text,"length_letters":a["letters"],"provenance":{"left_derivation":lf,"right_derivation":rf,"unique_roles":True,"source":"recursive semantic frame grammar"},"residual":front[:1],"independent_exact_audit":a,"mechanically_admitted":bool(ok and a["exact"] and a["letters"]>=39)}
   if row["mechanically_admitted"]:closures.append(row)
   elif len(w)<12:w.append(row)
 return {"status":"exhausted","stats":{"derivations":len(ps),"product_states":states,"rendered":len(w),"exact_accepted":len(closures),"longest_letters":max((x["length_letters"] for x in w),default=0)},"closures":closures,"diagnostic_witnesses":w,"config":{"recursive_semantic_frames":True,"unique_discourse_roles":True,"live_character_obligations":True,"no_seed_or_catalogue":True,"posthoc_reverse":False},"reader_gate":{"status":"not_triggered" if not closures else "human_blind_review_required","next_repair":"add a role-compatible transitive frame whose first residual character matches without reusing a discourse role"}}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.write_text(json.dumps(run(),indent=2)+'\n')
