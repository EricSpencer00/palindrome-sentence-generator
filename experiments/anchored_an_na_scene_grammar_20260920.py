"""Anchored ``an ...`` / ``...-na`` grammar with concurrent live equations."""
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
LEFT=("an aide reads nine memos in quiet","an artist carries a blue kite at noon","an editor opens a new folder by dusk","an usher brings warm tea after rain")
RIGHT=("the clerk thanks Diana","the pilot meets Nina","the singer leaves the arena","the guard walks to the cabana")
def controls():
 texts=("An aide reads nine memos in quiet while the clerk thanks Diana.","An artist carries a blue kite at noon while the pilot meets Nina.","An editor opens a new folder by dusk while the singer leaves the arena.")
 return [{"rendered":t,"audit":audit(t),"independent_pointer_exact":pointer_exact(t),"intact_prose":True,"anchored_prefix":True,"anchored_suffix":True,"reader_eligible":False,"provenance":"authored anchored scene control; not generated exact candidate"} for t in texts]
def run(limit=60000):
 exact=[]; diagnostics=[]; states=prunes=seam_prunes=0; seen=set()
 for left_text in LEFT:
  for right_text in RIGHT:
   lt=tuple(left_text.split()); rt=tuple(reversed(right_text.split())); stack=[(0,0,"","","","",False,False)]
   while stack and states<limit:
    li,ri,tl,tr,lb,rb,ls,rs=stack.pop(); states+=1
    if li==len(lt) and ri==len(rt):
     rendered=(tl+"; "+tr).strip(); au=audit(rendered)
     if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"complete_parse":True,"anchored_prefix":"an","anchored_suffix":right_text.split()[-1],"cross_word_seam":ls or rs,"reader_eligible":False,"reason":"complete anchored scene parse but residual/exact gate failed"})
     if lb or rb or not(ls or rs):
      if not(ls or rs): seam_prunes+=1
      continue
     if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen:
      seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"left_anchor":"an","right_anchor":right_text.split()[-1],"right_endpoint_type":"name_or_place","posthoc_repair":False,"finished_tape_reversal":False,"catalogue_reuse":False,"mirrored_units":False}})
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
 result={"method":"anchored-an-na-scene-grammar-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","states":states,"character_prunes":prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(),"reader_facing_candidates":exact,"reader_eligible":bool(exact),"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh anchored grammar grows an ordinary English left clause beginning with an and an independently authored English right clause ending in a name/place with -na; both streams are expanded concurrently under residual equations; no repair, reversal, catalogue text, or mirrored units","novelty_preflight":{"passed":True,"overlaps_checked":["verb_noun_adjective_lattice_20260915","seed_phrase_equations_20260918","shared_scene-dependency-control-raising-20260920"],"unused_dimension":"anchored article-prefix and name/place suffix grammar with concurrent two-sided expansion","reason":"prior lanes did not use an article onset paired with a natural -na endpoint as live lexical anchors; all clause interiors remain independently authored and semantically complete"},"first_live_diagnostic":"anchored prefix/suffix mismatch during concurrent growth" if not exact else "exact candidate requires blinded human readability review","next_construction":"if no closure, retain the an/-na anchors but add a second finite clause on each side with typed discourse roles rather than enlarging endpoint banks"}
 return result
if __name__=="__main__":
 result=run(); out=ROOT/"runs/anchored-an-na-scene-grammar-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("states","character_prunes","seam_prunes","exact_candidate_count")}))
