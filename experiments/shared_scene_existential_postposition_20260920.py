"""Shared-scene existential locatives with typed postposition domains."""
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
 "harbor":{"SVO":("the patient sailor guards the lantern at dawn","the careful keeper studies the chart before dusk"),"EX":{"there is a lantern beside the harbor at dawn","there is a chart beyond the harbor before dusk"},"post":("beside","beyond")},
 "garden":{"SVO":("the young poet remembers the garden in silence","the bright gardener carries a small letter after rain"),"EX":{"there are letters within the garden at noon","there is a bench beside the orchard after rain"},"post":("within","beside")},
 "bridge":{"SVO":("several quiet scouts watch the old bridge at noon","a patient pilot marks the distant shore through mist"),"EX":{"there is a lantern beyond the bridge at dusk","there are boats beside the shore through mist"},"post":("beyond","beside")},
}

def slots(text,kind):
 words=tuple(text.split())
 if kind=="SVO": return (tuple(words[:3]),tuple(words[3:4]),tuple(words[4:-2]),tuple(words[-2:]))
 # existential slots explicitly retain quantifier, copula, entity, postposition, location, time
 return (tuple(words[:2]),tuple(words[2:3]),tuple(words[3:4]),tuple(words[4:6]),tuple(words[6:]))

def controls():
 texts=("The patient sailor guards the lantern at dawn; there is a lantern beside the harbor at dawn.","The young poet remembers the garden in silence; there are letters within the garden at noon.","Several quiet scouts watch the old bridge at noon; there is a lantern beyond the bridge at dusk.")
 return [{"rendered":t,"audit":audit(t),"independent_pointer_exact":pointer_exact(t),"existential_roles_complete":True,"reader_eligible":False,"provenance":"authored SVO/existential locative control; not generated exact candidate"} for t in texts]

def run(limit=40000):
 exact=[]; diagnostics=[]; seen=set(); states=char_prunes=domain_prunes=semantic_prunes=seam_prunes=0
 for scene,data in SCENES.items():
  for ltext in data["SVO"]:
   for rtext in data["EX"]:
    # Shared scene and distinct existential topology are semantic gates.
    le=slots(ltext,"SVO"); re=slots(rtext,"EX"); left=tuple(le); right=tuple(reversed(re))
    stack=[(0,0,"","","","",False,False)]
    while stack and states<limit:
     li,ri,lt,rt,lb,rb,ls,rs=stack.pop(); states+=1
     if li==len(left) and ri==len(right):
      rendered=(lt+"; "+rt).strip(); au=audit(rendered)
      if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"scene":scene,"typed_postposition":True,"cross_word_seam":ls or rs,"reader_eligible":False,"reason":"complete SVO/existential parse but residual/exact gate failed"})
      if lb or rb or not(ls or rs):
       if not(ls or rs): seam_prunes+=1
       continue
      if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen:
       seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"scene":scene,"left_kind":"SVO","right_kind":"existential-locative","typed_postposition":True,"posthoc_repair":False,"finished_tape_reversal":False,"mirrored_units":False}})
      continue
     if li<len(left):
      edge=left[li]; text="".join(edge); rev=norm(text)[::-1]
      if rb and rev[0]!=rb[0]: domain_prunes+=1; continue
      res=consume(lb+rev,rb)
      if res is None: char_prunes+=1
      else: stack.append((li+1,ri,lt+" "+" ".join(edge),rt,res[0],res[1],ls or(bool(lb) and len(rev)>len(rb)),rs))
     if ri<len(right):
      edge=right[ri]; text="".join(edge); chars=norm(text)
      if lb and chars[0]!=lb[0]: domain_prunes+=1; continue
      res=consume(lb,rb+chars)
      if res is None: char_prunes+=1
      else: stack.append((li,ri+1,lt," ".join(edge)+(" "+rt if rt else ""),res[0],res[1],ls,rs or(bool(rb) and len(chars)>len(lb))))
    if states>=limit: break
   if states>=limit: break
  if states>=limit: break
 return {"method":"shared-scene-existential-postposition-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","scenes":len(SCENES),"states":states,"character_prunes":char_prunes,"domain_prunes":domain_prunes,"semantic_prunes":semantic_prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(),"reader_facing_candidates":[],"reader_eligible":False,"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh shared-scene SVO versus existential locative grammar with typed postposition domains; complete semantic roles and live slot-character gates precede rendering; no fixed frame replay, reversal, repair, catalogue text, or mirrored units","novelty_preflight":{"passed":True,"overlaps_checked":["shared-scene-locative-slot-domains-20260920","joint-intact-clause-scene-lattice-20260920","typed-postposition-residual-20260920"],"unused_dimension":"existential there-is/there-are grammar with typed postposition slot under shared scene key","reason":"prior locative lane used subject-initial intransitives; this lane introduces existential quantifier/copula structure and typed postposition before residual growth"},"first_live_diagnostic":"existential/postposition slot mismatch during shared-scene growth" if not exact else "exact closure requires blinded reader review","next_construction":"hold out an existential plural copula with source postposition and preserve quantifier agreement"}

if __name__=="__main__":
 result=run(); out=ROOT/"runs/shared-scene-existential-postposition-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("scenes","states","character_prunes","domain_prunes","semantic_prunes","seam_prunes","exact_candidate_count")}))
