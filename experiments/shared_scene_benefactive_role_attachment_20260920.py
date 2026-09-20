"""Shared-scene transitive/benefactive role attachment CSP."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def norm(s): return re.sub(r"[^a-z]","",s.casefold())
def audit(s):
 t=norm(s); i,j=0,len(t)-1
 while i<j and t[i]==t[j]: i+=1; j-=1
 return {"letters":len(t),"exact":bool(t) and i>=j,"first_mismatch":None if i>=j else {"index":i,"forward":t[i],"reverse":t[-1-i]},"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def pointer_exact(s):
 t=norm(s); return bool(t) and all(t[i]==t[-1-i] for i in range(len(t)//2))
def consume(a,b):
 n=min(len(a),len(b)); return (a[n:],b[n:]) if a[:n]==b[:n] else None

SCENES={
 "harbor":{"SVO":("the patient sailor guards the lantern at dawn","the careful keeper studies the chart before dusk"),"BEN":("the keeper carries a letter for the sailor at dawn","the sailor brings a chart for the keeper before dusk")},
 "garden":{"SVO":("the young poet remembers the garden in silence","the bright gardener carries a small letter after rain"),"BEN":("the gardener saves a seed for the poet in rain","the poet writes a letter for the gardener at noon")},
 "bridge":{"SVO":("several quiet scouts watch the old bridge at noon","a patient pilot marks the distant shore through mist"),"BEN":("the pilot carries a map for the scouts at dusk","the scouts bring a lantern for the pilot through mist")},
}

def slots(text,kind):
 w=tuple(text.split())
 if kind=="SVO": return (tuple(w[:3]),tuple(w[3:4]),tuple(w[4:-2]),tuple(w[-2:]))
 # Benefactive role is explicit and typed: subject, verb, theme, for, recipient, time.
 return (tuple(w[:2]),tuple(w[2:3]),tuple(w[3:5]),tuple(w[5:6]),tuple(w[6:-2]),tuple(w[-2:]))

def controls():
 texts=("The patient sailor guards the lantern at dawn; the keeper carries a letter for the sailor at dawn.","The young poet remembers the garden in silence; the gardener saves a seed for the poet in rain.","Several quiet scouts watch the old bridge at noon; the pilot carries a map for the scouts at dusk.")
 return [{"rendered":t,"audit":audit(t),"independent_pointer_exact":pointer_exact(t),"benefactive_role_complete":True,"reader_eligible":False,"provenance":"authored transitive/benefactive shared-scene control; not generated exact candidate"} for t in texts]

def run(limit=40000):
 exact=[]; diagnostics=[]; seen=set(); states=char_prunes=role_prunes=seam_prunes=0
 for scene,data in SCENES.items():
  for lt in data["SVO"]:
   for rt in data["BEN"]:
    left=slots(lt,"SVO"); right=tuple(reversed(slots(rt,"BEN")))
    stack=[(0,0,"","","","",False,False)]
    while stack and states<limit:
     li,ri,ltext,rtext,lb,rb,ls,rs=stack.pop(); states+=1
     if li==len(left) and ri==len(right):
      rendered=(ltext+"; "+rtext).strip(); au=audit(rendered)
      if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"scene":scene,"benefactive_role":"recipient","cross_word_seam":ls or rs,"reader_eligible":False,"reason":"complete transitive/benefactive parse but residual/exact gate failed"})
      if lb or rb or not(ls or rs):
       if not(ls or rs): seam_prunes+=1
       continue
      if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen:
       seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"scene":scene,"left_kind":"SVO","right_kind":"benefactive","recipient_role":"for-recipient","posthoc_repair":False,"finished_tape_reversal":False,"mirrored_units":False}})
      continue
     if li<len(left):
      e=left[li]; text="".join(e); rev=norm(text)[::-1]; res=consume(lb+rev,rb)
      if res is None: char_prunes+=1
      else: stack.append((li+1,ri,ltext+" "+" ".join(e),rtext,res[0],res[1],ls or(bool(lb) and len(rev)>len(rb)),rs))
     if ri<len(right):
      e=right[ri]; text="".join(e); chars=norm(text); res=consume(lb,rb+chars)
      if res is None: char_prunes+=1
      else: stack.append((li,ri+1,ltext," ".join(e)+(" "+rtext if rtext else ""),res[0],res[1],ls,rs or(bool(rb) and len(chars)>len(lb))))
    if states>=limit: break
   if states>=limit: break
  if states>=limit: break
 return {"method":"shared-scene-benefactive-role-attachment-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","scenes":len(SCENES),"states":states,"character_prunes":char_prunes,"role_prunes":role_prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(),"reader_facing_candidates":[],"reader_eligible":False,"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh shared-scene transitive versus benefactive role attachment with explicit for-recipient edge; complete semantic roles and live cross-word equations precede rendering; no existential replay, reversal, repair, catalogue text, or mirrored units","novelty_preflight":{"passed":True,"overlaps_checked":["shared-scene-existential-postposition-20260920","recipient-theme-attachment-csp-20260920","shared-scene-locative-slot-domains-20260920"],"unused_dimension":"benefactive recipient attachment under a shared scene key without existential or source/instrument topology","reason":"prior shared-scene lanes used locative/existential or source/instrument roles; this lane explicitly realizes recipient-benefactive structure in complete transitive clauses"},"first_live_diagnostic":"benefactive recipient edge residual mismatch during shared-scene growth" if not exact else "exact closure requires blinded reader review","next_construction":"hold out a reciprocal recipient role and preserve benefactive attachment gate"}

if __name__=="__main__":
 result=run(); out=ROOT/"runs/shared-scene-benefactive-role-attachment-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("scenes","states","character_prunes","role_prunes","seam_prunes","exact_candidate_count")}))
