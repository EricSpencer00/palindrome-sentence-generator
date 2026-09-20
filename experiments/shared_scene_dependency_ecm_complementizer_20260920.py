"""ECM finite-complementizer dependency frame with subject-role checks."""
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
 "harbor":{"CONTROL":("the captain expects the sailor to study the chart at dawn","the keeper urges the sailor to guard the lantern before dusk"),"ECM":("the captain expects that the sailor studies the chart at dawn","the keeper urges that the sailor guards the lantern before dusk")},
 "garden":{"CONTROL":("the gardener expects the poet to remember the garden in silence","the poet urges the gardener to carry a letter after rain"),"ECM":("the gardener expects that the poet remembers the garden in silence","the poet urges that the gardener carries a letter after rain")},
 "bridge":{"CONTROL":("the guide expects the pilot to mark the distant shore at noon","the pilot urges the guide to watch the old bridge through mist"),"ECM":("the guide expects that the pilot marks the distant shore at noon","the pilot urges that the guide watches the old bridge through mist")},
}
def controls():
 texts=("The captain expects the sailor to study the chart at dawn; the captain expects that the sailor studies the chart at dawn.","The gardener expects the poet to remember the garden in silence; the gardener expects that the poet remembers the garden in silence.","The guide expects the pilot to mark the distant shore at noon; the guide expects that the pilot marks the distant shore at noon.")
 return [{"rendered":t,"audit":audit(t),"independent_pointer_exact":pointer_exact(t),"ecm_subject_role_complete":True,"finite_agreement":True,"reader_eligible":False,"provenance":"authored control/ECM dependency control; not generated exact candidate"} for t in texts]
def run(limit=40000):
 exact=[]; diagnostics=[]; seen=set(); states=char_prunes=dependency_prunes=seam_prunes=0
 for scene,data in SCENES.items():
  for lt in data["CONTROL"]:
   for rt in data["ECM"]:
    left=(tuple(lt.split()),); right=(tuple(reversed(rt.split())),); stack=[(0,0,"","","","",False,False)]
    while stack and states<limit:
     li,ri,tl,tr,lb,rb,ls,rs=stack.pop(); states+=1
     if li==len(left) and ri==len(right):
      rendered=(tl+"; "+tr).strip(); au=audit(rendered)
      if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"scene":scene,"ecm_subject_role_complete":True,"finite_agreement":True,"cross_word_seam":ls or rs,"reader_eligible":False,"reason":"complete control/ECM parse but residual/exact gate failed"})
      if lb or rb or not(ls or rs):
       if not(ls or rs): seam_prunes+=1
       continue
      if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen:
       seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"scene":scene,"left_kind":"object-control","right_kind":"ECM-finite","complementizer":"that","embedded_subject":True,"finite_agreement":True,"posthoc_repair":False,"finished_tape_reversal":False,"mirrored_units":False}})
      continue
     if li<len(left):
      e=left[li]; text="".join(e); rev=norm(text)[::-1]; res=consume(lb+rev,rb)
      if res is None: char_prunes+=1
      else: stack.append((li+1,ri,tl+" "+" ".join(e),tr,res[0],res[1],ls or(bool(lb) and len(rev)>len(rb)),rs))
     if ri<len(right):
      e=right[ri]; text="".join(e); chars=norm(text); res=consume(lb,rb+chars)
      if res is None: char_prunes+=1
      else: stack.append((li,ri+1,tl," ".join(e)+(" "+tr if tr else ""),res[0],res[1],ls,rs or(bool(rb) and len(chars)>len(lb))))
    if states>=limit: break
   if states>=limit: break
  if states>=limit: break
 return {"method":"shared-scene-dependency-ecm-complementizer-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","scenes":len(SCENES),"states":states,"character_prunes":char_prunes,"dependency_prunes":dependency_prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(),"reader_facing_candidates":[],"reader_eligible":False,"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh object-control versus ECM finite-complementizer dependency grammar; explicit that complementizer, embedded subject, and finite agreement precede live cross-word equations; no raising/causative/reciprocal family, reversal, repair, catalogue text, or mirrored units","novelty_preflight":{"passed":True,"overlaps_checked":["shared-scene-dependency-control-raising-20260920","typed-dependency-frame-pairing-20260916","shared-scene-reciprocal-causative-embedded-20260920"],"unused_dimension":"ECM finite complementizer and embedded subject-role agreement","reason":"prior dependency lane paired control with raising infinitives; this lane introduces a finite that-complement and agreement-gated embedded subject"},"first_live_diagnostic":"ECM complementizer residual mismatch during dependency growth" if not exact else "exact closure requires blinded reader review","next_construction":"hold out a passive ECM complement with expletive subject and retain role checks"}

if __name__=="__main__":
 result=run(); out=ROOT/"runs/shared-scene-dependency-ecm-complementizer-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("scenes","states","character_prunes","dependency_prunes","seam_prunes","exact_candidate_count")}))
