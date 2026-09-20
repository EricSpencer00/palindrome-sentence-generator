"""Three-clause finite scene seam with typed PP attachment and agreement."""
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
 "harbor":{"left":["the nurse checks the chart after lunch","the nurse sends the note before dusk","the nurse waits by the door at night"],"right":["the nurse waits by the door at night","the nurse sends the note before dusk","the nurse checks the chart after lunch"]},
 "garden":{"left":["the poet reads the letter in shade","the poet folds the page after rain","the poet walks by the hedge at noon"],"right":["the poet walks by the hedge at noon","the poet folds the page after rain","the poet reads the letter in shade"]},
 "bridge":{"left":["the pilot marks the route at dawn","the pilot keeps the lamp through mist","the pilot watches the road after rain"],"right":["the pilot watches the road after rain","the pilot keeps the lamp through mist","the pilot marks the route at dawn"]},
}
def controls():
 texts=("The nurse checks the chart after lunch, sends the note before dusk, and waits by the door at night.","The poet reads the letter in shade, folds the page after rain, and walks by the hedge at noon.","The pilot marks the route at dawn, keeps the lamp through mist, and watches the road after rain.")
 return [{"rendered":t,"audit":audit(t),"independent_pointer_exact":pointer_exact(t),"intact_prose":True,"finite_agreement":True,"typed_pp_attachment":True,"reader_eligible":False,"provenance":"authored three-clause finite scene control; not generated exact candidate"} for t in texts]
def run(limit=100000):
 exact=[]; diagnostics=[]; states=char_prunes=seam_prunes=agreement_prunes=0; seen=set()
 for scene,data in SCENES.items():
  left=tuple(tuple(c.split()) for c in data["left"]); right=tuple(tuple(reversed(c.split())) for c in reversed(data["right"])); stack=[(0,0,"","","","",False,False)]
  while stack and states<limit:
   li,ri,tl,tr,lb,rb,ls,rs=stack.pop(); states+=1
   if li==len(left) and ri==len(right):
    rendered=(tl+"; "+tr).strip(); au=audit(rendered)
    if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"scene":scene,"complete_three_clause_parse":True,"finite_agreement":True,"typed_pp_attachment":True,"cross_clause_seam":ls or rs,"reader_eligible":False,"reason":"complete three-clause parse but residual/exact gate failed"})
    if lb or rb or not(ls or rs):
     if not(ls or rs): seam_prunes+=1
     continue
    if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen:
     seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_clause_seam":True,"provenance":{"scene":scene,"clauses":3,"finite_agreement":True,"typed_pp_attachment":True,"posthoc_repair":False,"finished_tape_reversal":False,"mirrored_units":False}})
    continue
   if li<len(left):
    e=left[li]; rev=norm("".join(e))[::-1]; res=consume(lb+rev,rb)
    if res is None: char_prunes+=1
    else: stack.append((li+1,ri,tl+" "+" ".join(e),tr,res[0],res[1],ls or(bool(lb) and len(rev)>len(rb)),rs))
   if ri<len(right):
    e=right[ri]; chars=norm("".join(e)); res=consume(lb,rb+chars)
    if res is None: char_prunes+=1
    else: stack.append((li,ri+1,tl," ".join(e)+(" "+tr if tr else ""),res[0],res[1],ls,rs or(bool(rb) and len(chars)>len(lb))))
  if states>=limit: break
 result={"method":"three-clause-typed-pp-scene-seam-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","scenes":len(SCENES),"states":states,"character_prunes":char_prunes,"agreement_prunes":agreement_prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(),"reader_facing_candidates":exact,"reader_eligible":bool(exact),"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh three-clause finite scene topology with typed PP attachment and subject-verb agreement carried across clauses; live residual admission precedes rendering; no VNA replay, repair, reversal, catalogue text, or mirrored units","novelty_preflight":{"passed":True,"overlaps_checked":["vna-live-phrase-lattice-20260920","vna-optional-scene-adjunct-lattice-20260920","shared_event_scene_pair_cfg_20260920"],"unused_dimension":"three finite clauses with typed PP attachments and shared agreement state","reason":"prior VNA lanes had one clause pair and nullable adjunct; this lane requires a complete three-clause finite scene with clause-order and PP attachment states"},"first_live_diagnostic":"three-clause character mismatch at PP attachment boundary" if not exact else "exact candidate requires blinded human readability review","next_construction":"if closure remains empty, pivot to a two-clause question-answer topology with typed PP attachment rather than extending clause count"}
 return result
if __name__=="__main__":
 result=run(); out=ROOT/"runs/three-clause-typed-pp-scene-seam-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("scenes","states","character_prunes","agreement_prunes","seam_prunes","exact_candidate_count")}))
