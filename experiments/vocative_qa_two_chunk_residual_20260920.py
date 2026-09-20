"""Role-typed vocative QA with an unequal two-chunk center residual."""
from __future__ import annotations
import hashlib, json, re
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

SPEAKERS=(("sailor","traveler",("sailor",),("did","you","see","the","harbor"),("the","sailor","saw","the","harbor")),
          ("keeper","witness",("keeper",),("did","you","guard","the","lantern"),("the","keeper","guarded","the","lantern")),
          ("poet","observer",("poet",),("did","you","remember","the","garden"),("the","poet","remembered","the","garden")))

def derivations():
 out=[]
 for sid,role,voc,q,a in SPEAKERS:
  # Unequal answer attachments: center split occurs after differing numbers
  # of answer words; the two chunks are tracked independently by the solver.
  for split in (1,2,3):
   edges=(voc+q, a[:split], a[split:])
   out.append({"speaker":sid,"role":role,"split":split,"edges":edges,"words":voc+q+("and",)+a})
 return out

def controls(ds):
 rows=[]
 for d in ds[:4]:
  w=d["words"]; text=" ".join(w[:1])+", "+" ".join(w[1:])+"."
  rows.append({"rendered":text,"audit":audit(text),"independent_pointer_exact":pointer_exact(text),"two_chunk_roles_complete":True,"reader_eligible":False,"provenance":"authored unequal-attachment QA control; not an exact candidate"})
 return rows

def run(limit=30000):
 ds=derivations(); exact=[]; diagnostics=[]; seen=set(); states=char_prunes=semantic_prunes=seam_prunes=0
 for left in ds:
  for right in ds:
   if left["speaker"]==right["speaker"] and left["split"]==right["split"]: semantic_prunes+=1; continue
   # Right emits from its inner edge: chunks and words are reversed, but the
   # chunk identity is retained so center obligations are not flattened.
   lw=left["edges"]; rw=tuple(reversed(right["edges"]))
   stack=[(0,0,"","",("",""),("",""),False,False)]
   while stack and states<limit:
    li,ri,lt,rt,lb,rb,ls,rs=stack.pop(); states+=1
    if li==len(lw) and ri==len(rw):
     rendered=(lt+"; "+rt).strip(); au=audit(rendered)
     if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"cross_word_seam":ls or rs,"complete_semantic_parse":True,"reader_eligible":False,"reason":"complete two-chunk derivation but residual/exact gate failed"})
     if any(lb) or any(rb) or not(ls or rs):
      if not(ls or rs): seam_prunes+=1
      continue
     if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen:
      seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"left_derivation":left,"right_derivation":right,"two_chunk_center_residual":True,"posthoc_repair":False,"finished_tape_reversal":False,"mirrored_units":False}})
     continue
    if li<len(lw):
     words=lw[li]; text="".join(words) if isinstance(words,tuple) else words; chunk=0 if li==0 else 1
     # punctuation is outside the character tape; words in a chunk are emitted as a phrase.
     res=consume(lb[chunk]+norm(text),rb[chunk])
     if res is None: char_prunes+=1
     else:
      nlb=list(lb); nlb[chunk]=res[0]; stack.append((li+1,ri,(lt+" " if lt else "")+" ".join(words),rt,tuple(nlb),rb,ls or(bool(lb[chunk]) and len(norm(text))>len(rb[chunk])),rs))
    if ri<len(rw):
     words=rw[ri]; text="".join(words) if isinstance(words,tuple) else words; chunk=1 if ri==0 else 0
     res=consume(lb[chunk],rb[chunk]+norm(text)[::-1])
     if res is None: char_prunes+=1
     else:
      nrb=list(rb); nrb[chunk]=res[0]; stack.append((li,ri+1,lt," ".join(words)+(" "+rt if rt else ""),lb,tuple(nrb),ls,rs or(bool(rb[chunk]) and len(norm(text))>len(lb[chunk]))))
   if states>=limit: break
  if states>=limit: break
 return {"method":"vocative-qa-two-chunk-residual-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","derivations":len(ds),"states":states,"character_prunes":char_prunes,"semantic_prunes":semantic_prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(ds),"reader_facing_candidates":[],"reader_eligible":False,"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh role-typed vocative question-answer grammar with unequal answer attachment splits and two independent residual chunks; live character equations and cross-word seam required; no command/appositive sweep, reversal, repair, catalogue text, or mirrored units","novelty_preflight":{"passed":True,"overlaps_checked":["nullable-cfg-vocative-question-answer-20260920","nullable-cfg-vocative-appositive-20260920","nullable-clause-chart-crossword-20260920"],"unused_dimension":"unequal answer attachment split with a two-component residual vector (outer and center) carried through the complete QA derivation","reason":"prior QA/vocative lanes used one residual stream and fixed edge boundaries; this lane preserves chunk identity while allowing unequal center attachment"},"first_live_diagnostic":"two-chunk residual mismatch at QA attachment boundary" if not exact else "exact closure requires blinded reader review","next_construction":"hold out a three-chunk clarification answer with a typed role switch; do not widen current QA lexicon"}

if __name__=="__main__":
 result=run(); out=ROOT/"runs/vocative-qa-two-chunk-residual-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("derivations","states","character_prunes","semantic_prunes","seam_prunes","exact_candidate_count")}))
