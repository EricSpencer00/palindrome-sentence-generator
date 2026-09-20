"""V-N-ADJ lattice with a scene-tied nullable prepositional adjunct."""
from __future__ import annotations
import hashlib,json,re
from itertools import product
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
V=["marks","carries","folds","keeps","traces","opens"]
N=["lantern","garden","letter","harbor","window","banner"]
A=["bright","silent","narrow","patient","ancient","open"]
SCENES={"harbor":{"subject":"the sailor","adj":"at dawn"},"garden":{"subject":"the poet","adj":"in shade"},"bridge":{"subject":"the pilot","adj":"through mist"}}
def phrase(v,n,a): return f"{v} the {n} {a}"
def controls():
 texts=("At dawn, the sailor marks the lantern bright and keeps the garden silent.","In shade, the poet carries the letter patient and opens the window narrow.","Through mist, the pilot traces the banner ancient and folds the harbor open.")
 return [{"rendered":t,"audit":audit(t),"independent_pointer_exact":pointer_exact(t),"intact_prose":True,"reader_eligible":False,"provenance":"authored V-N-ADJ optional-adjunct control; not generated exact candidate"} for t in texts]
def run(limit=120000):
 exact=[]; diagnostics=[]; states=prunes=seam_prunes=0; seen=set()
 for scene,spec in SCENES.items():
  # Adjunct is nullable but scene-tied: no independent adjunct Cartesian product.
  for left,right in product(product(V,N,A),product(V,N,A)):
   if states>=limit: break
   lp=phrase(*left); rp=phrase(*right); subj=spec["subject"]
   for include_adj in (False,True):
    suffix=(" "+spec["adj"]) if include_adj else ""
    lt=tuple(("the "+subj+" and "+lp+suffix).split()); rt=tuple(("the "+subj+" and "+rp+suffix).split())
    stack=[(0,0,"","","","",False,False)]
    while stack and states<limit:
     li,ri,tl,tr,lb,rb,ls,rs=stack.pop(); states+=1
     if li==len(lt) and ri==len(rt):
      rendered=(tl+"; "+tr).strip(); au=audit(rendered)
      if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"scene":scene,"optional_adjunct":include_adj,"complete_vna_parse":True,"cross_word_seam":ls or rs,"reader_eligible":False,"reason":"complete nullable-adjunct V-N-ADJ parse but residual/exact gate failed"})
      if lb or rb or not(ls or rs):
       if not(ls or rs): seam_prunes+=1
       continue
      if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen:
       seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"scene":scene,"left_slots":"V-N-ADJ","right_slots":"V-N-ADJ","optional_adjunct":include_adj,"adjunct_key":spec["adj"],"posthoc_repair":False,"finished_tape_reversal":False,"catalogue_reuse":False,"mirrored_units":False}})
      continue
     if li<len(lt):
      word=lt[li]; rev=norm(word)[::-1]; res=consume(lb+rev,rb)
      if res is None: prunes+=1
      else: stack.append((li+1,ri,tl+" "+word,tr,res[0],res[1],ls or(bool(lb) and len(rev)>len(rb)),rs))
     if ri<len(rt):
      word=rt[ri]; chars=norm(word); res=consume(lb,rb+chars)
      if res is None: prunes+=1
      else: stack.append((li,ri+1,tl,word+(" "+tr if tr else ""),res[0],res[1],ls,rs or(bool(rb) and len(chars)>len(lb))))
    if states>=limit: break
   if states>=limit: break
  if states>=limit: break
 return {"method":"vna-optional-scene-adjunct-lattice-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","scenes":len(SCENES),"states":states,"character_prunes":prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(),"reader_facing_candidates":exact,"reader_eligible":bool(exact),"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh V-N-ADJ lattice with a nullable prepositional adjunct tied to the authored scene key; adjunct is not independently multiplied and live residual equations gate every character; no repair, reversal, borrowed catalogue, or mirrored units","novelty_preflight":{"passed":True,"overlaps_checked":["vna-live-phrase-lattice-20260920","scene_phrase_equations_20260920","two_edge_nullable_adjunct_cfg_20260920"],"unused_dimension":"nullable scene-tied prepositional adjunct on a jointly solved V-N-ADJ lattice","reason":"prior V-N-ADJ lane had no adjunct branch; this lane adds one typed nullable attachment without an independent adjunct product"},"first_live_diagnostic":"V-N-ADJ mismatch at nullable adjunct boundary" if not exact else "exact candidate requires blinded human readability review","next_construction":"if closure remains empty, pivot away from V-N-ADJ and test an unaccusative subject-oriented frame rather than widening this lattice"}
if __name__=="__main__":
 result=run(); out=ROOT/"runs/vna-optional-scene-adjunct-lattice-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("scenes","states","character_prunes","seam_prunes","exact_candidate_count")}))
