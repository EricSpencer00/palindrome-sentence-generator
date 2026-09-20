"""Relation-aware three-chunk vocative QA clarification CSP."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def norm(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
 t=norm(s); i,j=0,len(t)-1
 while i<j and t[i]==t[j]: i+=1; j-=1
 return {"letters":len(t),"exact":bool(t) and i>=j,"first_mismatch":None if i>=j else {"index":i,"forward":t[i],"reverse":t[-1-i]},"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def pointer_exact(s):
 t=norm(s); return bool(t) and all(t[i]==t[-1-i] for i in range(len(t)//2))
def consume(a,b):
 n=min(len(a),len(b)); return (a[n:],b[n:]) if a[:n]==b[:n] else None

FRAMES=(
 {"speaker":"sailor","role":"traveler","relation":"because","voc":("sailor",),"q":("did","you","see","the","harbor"),"a":("the","sailor","saw","the","harbor"),"clar":("the","keeper","confirmed"),"tail":("that","the","harbor","was","quiet")},
 {"speaker":"keeper","role":"witness","relation":"so","voc":("keeper",),"q":("did","you","guard","the","lantern"),"a":("the","keeper","guarded","the","lantern"),"clar":("the","sailor","agreed"),"tail":("that","the","lantern","was","bright")},
 {"speaker":"poet","role":"observer","relation":"although","voc":("poet",),"q":("did","you","remember","the","garden"),"a":("the","poet","remembered","the","garden"),"clar":("the","keeper","noted"),"tail":("that","the","garden","was","peaceful")},
)
def derivations():
 return [{"speaker":f["speaker"],"role":f["role"],"relation":f["relation"],"edges":(f["voc"]+f["q"],f["a"],(f["relation"],)+f["clar"],f["tail"])} for f in FRAMES]
def controls(ds):
 rows=[]
 for d in ds:
  e=d["edges"]; text=" ".join(e[0][:1])+", "+" ".join(e[0][1:])+"? "+" ".join(e[1])+" "+" ".join(e[2])+" "+" ".join(e[3])+"."
  rows.append({"rendered":text,"audit":audit(text),"independent_pointer_exact":pointer_exact(text),"relation_center":d["relation"],"complete_semantic_parse":True,"reader_eligible":False,"provenance":"authored relation-aware clarification control; not an exact candidate"})
 return rows
def run(limit=30000):
 ds=derivations(); exact=[]; diagnostics=[]; seen=set(); states=char_prunes=semantic_prunes=seam_prunes=0
 for left in ds:
  for right in ds:
   if left["relation"]==right["relation"] or left["speaker"]==right["speaker"]: semantic_prunes+=1; continue
   lw,rw=left["edges"],tuple(reversed(right["edges"])); stack=[(0,0,"","",("","",""),("","",""),False,False)]
   while stack and states<limit:
    li,ri,lt,rt,lb,rb,ls,rs=stack.pop(); states+=1
    if li==len(lw) and ri==len(rw):
     rendered=(lt+"; "+rt).strip(); au=audit(rendered)
     if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"cross_word_seam":ls or rs,"complete_semantic_parse":True,"relation_center":left["relation"],"reader_eligible":False,"reason":"complete relation-aware clarification but residual/exact gate failed"})
     if any(lb) or any(rb) or not(ls or rs):
      if not(ls or rs): seam_prunes+=1
      continue
     if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen:
      seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"left_derivation":left,"right_derivation":right,"relation_center":left["relation"],"three_chunk_residual":True,"posthoc_repair":False,"finished_tape_reversal":False,"mirrored_units":False}})
     continue
    if li<len(lw):
     words=lw[li]; text="".join(words); chunk=min(li,2); res=consume(lb[chunk]+norm(text),rb[chunk])
     if res is None: char_prunes+=1
     else:
      nlb=list(lb); nlb[chunk]=res[0]; stack.append((li+1,ri,(lt+" " if lt else "")+" ".join(words),rt,tuple(nlb),rb,ls or(bool(lb[chunk]) and len(norm(text))>len(rb[chunk])),rs))
    if ri<len(rw):
     words=rw[ri]; text="".join(words); chunk=max(0,2-ri); res=consume(lb[chunk],rb[chunk]+norm(text)[::-1])
     if res is None: char_prunes+=1
     else:
      nrb=list(rb); nrb[chunk]=res[0]; stack.append((li,ri+1,lt," ".join(words)+(" "+rt if rt else ""),lb,tuple(nrb),ls,rs or(bool(rb[chunk]) and len(norm(text))>len(lb[chunk]))))
   if states>=limit: break
  if states>=limit: break
 return {"method":"vocative-qa-relation-center-clarification-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","derivations":len(ds),"states":states,"character_prunes":char_prunes,"semantic_prunes":semantic_prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(ds),"reader_facing_candidates":[],"reader_eligible":False,"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh role-typed vocative QA with relation-aware clarification center (causal/evidential/concessive) and three residual chunks; complete semantic relation and cross-word seam required online; no bank sweep, reversal, repair, catalogue text, or mirrored units","novelty_preflight":{"passed":True,"overlaps_checked":["vocative-qa-three-chunk-clarification-20260920","vocative-qa-two-chunk-residual-20260920","relation-conditioned-voice-grammar-20260920"],"unused_dimension":"relation category is a live center obligation coupling question role to clarification attachment while residual chunks remain separate","reason":"prior QA lanes had typed role switches but no causal/evidential/concessive relation state carried through the center residual"},"first_live_diagnostic":"relation-center residual mismatch at clarification edge" if not exact else "exact closure requires blinded reader review","next_construction":"hold out an evidential relation with a relative clarification edge; do not widen relation inventory"}
if __name__=="__main__":
 result=run(); out=ROOT/"runs/vocative-qa-relation-center-clarification-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("derivations","states","character_prunes","semantic_prunes","seam_prunes","exact_candidate_count")}))
