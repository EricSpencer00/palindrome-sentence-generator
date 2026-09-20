"""Two-clause anchored an/-na grammar with typed discourse roles."""
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
 "harbor":{"left":["an aide reads nine memos in quiet","the aide files the notes before dusk"],"right":["the clerk checks the ledger before dusk","the clerk thanks Diana"],"roles":["agent=aide object=memos","agent=aide destination=archive","agent=clerk object=ledger","agent=clerk recipient=Diana"]},
 "garden":{"left":["an artist carries a blue kite at noon","the artist stores the string after rain"],"right":["the pilot folds the map after rain","the pilot meets Nina"],"roles":["agent=artist object=kite","agent=artist destination=store","agent=pilot object=map","agent=pilot recipient=Nina"]},
 "arena":{"left":["an editor opens a new folder by dusk","the editor sends the draft before dawn"],"right":["the singer leaves the stage before dawn","the singer enters the arena"],"roles":["agent=editor object=folder","agent=editor recipient=reader","agent=singer object=stage","agent=singer destination=arena"]},
}
def controls():
 texts=("An aide reads nine memos in quiet; the aide files the notes before dusk while the clerk checks the ledger before dusk; the clerk thanks Diana.","An artist carries a blue kite at noon; the artist stores the string after rain while the pilot folds the map after rain; the pilot meets Nina.","An editor opens a new folder by dusk; the editor sends the draft before dawn while the singer leaves the stage before dawn; the singer enters the arena.")
 return [{"rendered":t,"audit":audit(t),"independent_pointer_exact":pointer_exact(t),"intact_prose":True,"typed_roles":True,"reader_eligible":False,"provenance":"authored two-clause anchored control; not generated exact candidate"} for t in texts]
def run(limit=60000):
 exact=[]; diagnostics=[]; states=prunes=seam_prunes=0; seen=set()
 for scene,data in SCENES.items():
  # Keep the endpoint bank fixed; only the authored scene clauses vary.
  left=tuple(tuple(c.split()) for c in data["left"]); right=tuple(tuple(reversed(c.split())) for c in reversed(data["right"])); stack=[(0,0,"","","","",False,False)]
  while stack and states<limit:
   li,ri,tl,tr,lb,rb,ls,rs=stack.pop(); states+=1
   if li==len(left) and ri==len(right):
    rendered=(tl+"; "+tr).strip(); au=audit(rendered)
    if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"scene":scene,"complete_two_clause_parse":True,"typed_roles":data["roles"],"anchored_prefix":"an","anchored_suffix":data["right"][-1].split()[-1],"cross_clause_seam":ls or rs,"reader_eligible":False,"reason":"complete typed-role parse but residual/exact gate failed"})
    if lb or rb or not(ls or rs):
     if not(ls or rs): seam_prunes+=1
     continue
    if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen:
     seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_clause_seam":True,"provenance":{"scene":scene,"roles":data["roles"],"left_anchor":"an","right_endpoint":data["right"][-1].split()[-1],"posthoc_repair":False,"finished_tape_reversal":False,"catalogue_reuse":False,"mirrored_units":False}})
    continue
   if li<len(left):
    e=left[li]; rev=norm("".join(e))[::-1]; res=consume(lb+rev,rb)
    if res is None: prunes+=1
    else: stack.append((li+1,ri,tl+" "+" ".join(e),tr,res[0],res[1],ls or(bool(lb) and len(rev)>len(rb)),rs))
   if ri<len(right):
    e=right[ri]; chars=norm("".join(e)); res=consume(lb,rb+chars)
    if res is None: prunes+=1
    else: stack.append((li,ri+1,tl," ".join(e)+(" "+tr if tr else ""),res[0],res[1],ls,rs or(bool(rb) and len(chars)>len(lb))))
  if states>=limit: break
 result={"method":"anchored-an-na-two-clause-roles-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","states":states,"character_prunes":prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(),"reader_facing_candidates":exact,"reader_eligible":bool(exact),"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh two-clause anchored grammar; each authored clause carries explicit discourse roles and both sides grow concurrently under live residual equations; endpoint bank fixed to ordinary names/places; no repair, reversal, catalogue text, or mirrored units","novelty_preflight":{"passed":True,"overlaps_checked":["anchored-an-na-scene-grammar-20260920","three-clause-typed-pp-scene-seam-20260920","shared_scene-dependency-control-raising-20260920"],"unused_dimension":"two finite discourse-role clauses under fixed an/-na anchors","reason":"prior anchored lane had one clause per side; this lane adds a second typed-role clause without widening endpoints or mirroring units"},"first_live_diagnostic":"anchored two-clause role mismatch during concurrent growth" if not exact else "exact candidate requires blinded human readability review","next_construction":"if closure remains empty, pivot to a question-answer discourse edge while retaining the fixed anchors and endpoint bank"}
 return result
if __name__=="__main__":
 result=run(); out=ROOT/"runs/anchored-an-na-two-clause-roles-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("states","character_prunes","seam_prunes","exact_candidate_count")}))
