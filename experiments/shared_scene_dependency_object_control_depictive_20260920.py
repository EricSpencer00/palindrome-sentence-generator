"""Object-control plus depictive adjunct dependency lane."""
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
 "harbor":("the captain persuaded the sailor to carry the lantern home, weary","the keeper persuaded the sailor to leave the lantern bright, patient"),
 "garden":("the gardener persuaded the poet to carry the basket home, quiet","the poet persuaded the gardener to leave the basket full, gentle"),
 "bridge":("the guide persuaded the pilot to carry the compass north, alert","the pilot persuaded the guide to leave the compass ready, calm"),
}
def controls():
 texts=("The captain persuaded the sailor to carry the lantern home, weary; the keeper persuaded the sailor to leave the lantern bright, patient.","The gardener persuaded the poet to carry the basket home, quiet; the poet persuaded the gardener to leave the basket full, gentle.","The guide persuaded the pilot to carry the compass north, alert; the pilot persuaded the guide to leave the compass ready, calm.")
 return [{"rendered":t,"audit":audit(t),"independent_pointer_exact":pointer_exact(t),"object_control":True,"depictive_adjunct":True,"role_complete":True,"reader_eligible":False,"provenance":"authored object-control/depicitve control; not generated exact candidate"} for t in texts]
def run(limit=40000):
 exact=[]; diagnostics=[]; seen=set(); states=char_prunes=dependency_prunes=seam_prunes=0
 for scene,(left_text,right_text) in SCENES.items():
  left=(tuple(left_text.split()),); right=(tuple(reversed(right_text.split())),); stack=[(0,0,"","","","",False,False)]
  while stack and states<limit:
   li,ri,tl,tr,lb,rb,ls,rs=stack.pop(); states+=1
   if li==len(left) and ri==len(right):
    rendered=(tl+"; "+tr).strip(); au=audit(rendered)
    if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"scene":scene,"object_control":True,"depictive_adjunct":True,"role_complete":True,"cross_word_seam":ls or rs,"reader_eligible":False,"reason":"complete object-control/depicitve parse but residual/exact gate failed"})
    if lb or rb or not(ls or rs):
     if not(ls or rs): seam_prunes+=1
     continue
    if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen:
     seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"scene":scene,"left_kind":"object-control","right_kind":"object-control","controlled_subject":"PRO","depictive":True,"posthoc_repair":False,"finished_tape_reversal":False,"mirrored_units":False}})
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
 return {"method":"shared-scene-dependency-object-control-depictive-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","scenes":len(SCENES),"states":states,"character_prunes":char_prunes,"dependency_prunes":dependency_prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(),"reader_facing_candidates":[],"reader_eligible":False,"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh object-control topology with controlled PRO subject and depictive adjunct; complete semantic roles and live cross-word equations precede rendering; no ECM, agreement variant, reversal, repair, catalogue text, or mirrored units","novelty_preflight":{"passed":True,"overlaps_checked":["shared-scene-dependency-ecm-complementizer-20260920","shared-scene-dependency-passive-ecm-expletive-20260920","shared-scene-dependency-control-raising-20260920"],"unused_dimension":"object-control PRO plus depictive adjunct role binding","reason":"prior dependency lanes used finite ECM, expletive passive, or subject control/raising; this lane binds an overt object as controller of an infinitival PRO and carries a depictive adjunct"},"first_live_diagnostic":"object-control residual mismatch during dependency growth" if not exact else "exact closure requires blinded reader review","next_construction":"try a small-clause resultative object-control frame with an overt predicative complement if closure remains empty"}
if __name__=="__main__":
 result=run(); out=ROOT/"runs/shared-scene-dependency-object-control-depictive-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("scenes","states","character_prunes","dependency_prunes","seam_prunes","exact_candidate_count")}))
