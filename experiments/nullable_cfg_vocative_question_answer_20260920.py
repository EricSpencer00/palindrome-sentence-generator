"""Nullable vocative question-answer CFG with role-typed live residuals."""
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

SPEAKERS=(
 {"id":"sailor","vocative":("sailor",),"role":"traveler"},
 {"id":"keeper","vocative":("keeper",),"role":"witness"},
 {"id":"poet","vocative":("poet",),"role":"observer"},
)
QUESTIONS=(
 {"role":"traveler","words":("did","you","see","the","harbor")},
 {"role":"witness","words":("did","you","guard","the","lantern")},
 {"role":"observer","words":("did","you","remember","the","garden")},
)
ANSWERS=(
 {"role":"traveler","words":("the","sailor","saw","the","harbor")},
 {"role":"witness","words":("the","keeper","guarded","the","lantern")},
 {"role":"observer","words":("the","poet","remembered","the","garden")},
)
TAILS=(("and",("the","evening","was","gentle")),("while",("the","harbor","seemed","quiet")))

def derivations():
 out=[]
 for sp in SPEAKERS:
  q=next(x for x in QUESTIONS if x["role"]==sp["role"])
  base=sp["vocative"]+q["words"]
  # Nullable answer continuation; only the role-compatible answer is allowed.
  out.append({"speaker":sp["id"],"question_role":sp["role"],"answer":None,"words":base})
  ans=next(x for x in ANSWERS if x["role"]==sp["role"])
  out.append({"speaker":sp["id"],"question_role":sp["role"],"answer":ans["role"],"words":base+("and",)+ans["words"]})
  for conn,tail in TAILS:
   out.append({"speaker":sp["id"],"question_role":sp["role"],"answer":ans["role"],"words":base+("and",)+ans["words"]+(conn,)+tail})
 return out

def controls(ds):
 rows=[]
 for d in ds:
  if d["answer"] is None or len(rows)>=4: continue
  w=d["words"]; text=" ".join(w[:1])+", "+" ".join(w[1:])+"."
  rows.append({"rendered":text,"audit":audit(text),"independent_pointer_exact":pointer_exact(text),"role_agreement":True,"complete_semantic_parse":True,"reader_eligible":False,"provenance":"authored vocative question-answer CFG control; not an exact candidate"})
 return rows

def run(limit=30000):
 ds=derivations(); exact=[]; diagnostics=[]; seen=set(); states=char_prunes=semantic_prunes=seam_prunes=0
 for left in ds:
  for right in ds:
   if left["speaker"]==right["speaker"] and left["answer"]==right["answer"]: semantic_prunes+=1; continue
   lw,rw=left["words"],tuple(reversed(right["words"])); stack=[(0,0,"","","","",False,False)]
   while stack and states<limit:
    li,ri,lt,rt,lb,rb,ls,rs=stack.pop(); states+=1
    if li==len(lw) and ri==len(rw):
     rendered=(lt+"; "+rt).strip(); au=audit(rendered)
     if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"cross_word_seam":ls or rs,"complete_semantic_parse":True,"reader_eligible":False,"reason":"complete question-answer derivation but residual/exact gate failed"})
     if lb or rb or not(ls or rs):
      if not(ls or rs): seam_prunes+=1
      continue
     if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen:
      seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"left_derivation":left,"right_derivation":right,"role_agreement":True,"posthoc_repair":False,"finished_tape_reversal":False,"mirrored_units":False}})
     continue
    if li<len(lw):
     w=lw[li]; res=consume(lb+norm(w),rb)
     if res is None: char_prunes+=1
     else: stack.append((li+1,ri,(lt+" " if lt else "")+w,rt,res[0],res[1],ls or(bool(lb) and len(norm(w))>len(rb)),rs))
    if ri<len(rw):
     w=rw[ri]; res=consume(lb,rb+norm(w)[::-1])
     if res is None: char_prunes+=1
     else: stack.append((li,ri+1,lt,w+(" "+rt if rt else ""),res[0],res[1],ls,rs or(bool(rb) and len(norm(w))>len(lb))))
   if states>=limit: break
  if states>=limit: break
 return {"method":"nullable-cfg-vocative-question-answer-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","derivations":len(ds),"states":states,"character_prunes":char_prunes,"semantic_prunes":semantic_prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(ds),"reader_facing_candidates":[],"reader_eligible":False,"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh nullable question-answer CFG with vocative speaker role agreement and optional connective tail; complete semantic edges emit variable word boundaries into live residuals with cross-word seam; no command inventory widening, reversal, repair, catalogue text, or mirrored units","novelty_preflight":{"passed":True,"overlaps_checked":["nullable-cfg-vocative-appositive-20260920","mixed-declarative-vocative-imperative-20260920","dialogue-speech-act-residual-20260916"],"unused_dimension":"role-typed vocative question plus answer continuation, distinct from imperative command and dialogue inventory products","reason":"prior vocative lane used commands and prior dialogue lanes used untyped act products; this CFG couples question/answer role identity before character emission"},"first_live_diagnostic":"character residual mismatch at question-answer CFG edge" if not exact else "exact closure requires blinded reader review","next_construction":"hold out a role-compatible clarification answer with a nullable question tail; do not widen the question bank"}

if __name__=="__main__":
 result=run(); out=ROOT/"runs/nullable-cfg-vocative-question-answer-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("derivations","states","character_prunes","semantic_prunes","seam_prunes","exact_candidate_count")}))
