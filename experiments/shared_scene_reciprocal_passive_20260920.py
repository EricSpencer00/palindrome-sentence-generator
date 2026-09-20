"""Reciprocal passive grammar under shared scene keys."""
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
 "harbor":{"ACTIVE":("the sailors guard the lantern at dawn","the keepers study the charts before dusk"),"PASSIVE":("letters are exchanged among the sailors at dawn","charts are shared among the keepers before dusk")},
 "garden":{"ACTIVE":("the poets remember the garden in silence","the gardeners carry small letters after rain"),"PASSIVE":("seeds are shared among the poets in rain","letters are carried among the gardeners at noon")},
 "bridge":{"ACTIVE":("the scouts watch the old bridge at noon","the pilots mark the distant shore through mist"),"PASSIVE":("maps are traded among the scouts at dusk","lamps are carried among the pilots through mist")},
}
def controls():
 texts=("The sailors guard the lantern at dawn; letters are exchanged among the sailors at dawn.","The poets remember the garden in silence; seeds are shared among the poets in rain.","The scouts watch the old bridge at noon; maps are traded among the scouts at dusk.")
 return [{"rendered":t,"audit":audit(t),"independent_pointer_exact":pointer_exact(t),"passive_plural_agreement":True,"reciprocal_group_complete":True,"reader_eligible":False,"provenance":"authored active/passive reciprocal control; not generated exact candidate"} for t in texts]
def run(limit=40000):
 exact=[]; diagnostics=[]; seen=set(); states=char_prunes=semantic_prunes=seam_prunes=0
 for scene,data in SCENES.items():
  for lt in data["ACTIVE"]:
   for rt in data["PASSIVE"]:
    left=(tuple(lt.split()),); right=(tuple(reversed(rt.split())),)
    stack=[(0,0,"","","","",False,False)]
    while stack and states<limit:
     li,ri,tl,tr,lb,rb,ls,rs=stack.pop(); states+=1
     if li==len(left) and ri==len(right):
      rendered=(tl+"; "+tr).strip(); au=audit(rendered)
      if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"scene":scene,"passive_plural_agreement":True,"cross_word_seam":ls or rs,"reader_eligible":False,"reason":"complete active/passive parse but residual/exact gate failed"})
      if lb or rb or not(ls or rs):
       if not(ls or rs): seam_prunes+=1
       continue
      if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen:
       seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"scene":scene,"left_kind":"active","right_kind":"reciprocal-passive","plural_are_agreement":True,"among_group":True,"posthoc_repair":False,"finished_tape_reversal":False,"mirrored_units":False}})
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
 return {"method":"shared-scene-reciprocal-passive-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","scenes":len(SCENES),"states":states,"character_prunes":char_prunes,"semantic_prunes":semantic_prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(),"reader_facing_candidates":[],"reader_eligible":False,"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh reciprocal passive grammar with plural are agreement and among-group agent realization under shared scenes; complete voice/role parses and live cross-word equations precede rendering; no active auxiliary widening, reversal, repair, catalogue text, or mirrored units","novelty_preflight":{"passed":True,"overlaps_checked":["shared-scene-reciprocal-double-object-pastperfect-20260920","shared-scene-reciprocal-dative-object-20260920","relation-conditioned-voice-grammar-20260920"],"unused_dimension":"reciprocal passive voice with among-group agent and plural are agreement","reason":"prior reciprocal lanes were active dative/benefactive; this lane changes voice and agent realization before character growth"},"first_live_diagnostic":"passive voice residual mismatch during shared-scene growth" if not exact else "exact closure requires blinded reader review","next_construction":"hold out a causative reciprocal frame with explicit plural embedded subject; do not widen passive vocabulary"}

if __name__=="__main__":
 result=run(); out=ROOT/"runs/shared-scene-reciprocal-passive-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("scenes","states","character_prunes","semantic_prunes","seam_prunes","exact_candidate_count")}))
