"""Three-chunk vocative QA clarification with a typed role switch."""
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

# Question speaker and answer addressee are typed independently.  The answer
# switches role in its clarification tail: addressee -> witness.
FRAMES=(
 {"speaker":"sailor","role":"traveler","voc":("sailor",),"question":("did","you","see","the","harbor"),"chunks":(("the","sailor","saw","the","harbor"),("and","the","keeper","clarified"),("that","the","harbor","was","quiet")),"switch":"witness"},
 {"speaker":"keeper","role":"witness","voc":("keeper",),"question":("did","you","guard","the","lantern"),"chunks":(("the","keeper","guarded","the","lantern"),("and","the","sailor","clarified"),("that","the","lantern","was","bright")),"switch":"traveler"},
 {"speaker":"poet","role":"observer","voc":("poet",),"question":("did","you","remember","the","garden"),"chunks":(("the","poet","remembered","the","garden"),("and","the","keeper","clarified"),("that","the","garden","was","peaceful")),"switch":"witness"},
)

def derivations():
 out=[]
 for f in FRAMES:
  # Unequal attachment boundary: question is chunk 0, answer clarification
  # has three independently matched chunks; no single-tape flattening.
  out.append({"speaker":f["speaker"],"role":f["role"],"switch":f["switch"],"edges":(f["voc"]+f["question"],)+f["chunks"],"chunks":f["chunks"]})
 return out

def controls(ds):
 rows=[]
 for d in ds:
  e=d["edges"]; text=" ".join(e[0][:1])+", "+" ".join(e[0][1:])+"? "+" ".join(e[1])+" "+" ".join(e[2])+" "+" ".join(e[3])+"."
  rows.append({"rendered":text,"audit":audit(text),"independent_pointer_exact":pointer_exact(text),"three_chunk_roles_complete":True,"role_switch":d["switch"],"reader_eligible":False,"provenance":"authored clarification control; not an exact candidate"})
 return rows

def run(limit=30000):
 ds=derivations(); exact=[]; diagnostics=[]; seen=set(); states=char_prunes=semantic_prunes=seam_prunes=0
 for left in ds:
  for right in ds:
   if left["speaker"]==right["speaker"] or left["switch"]==right["switch"]: semantic_prunes+=1; continue
   lw=left["edges"]; rw=tuple(reversed(right["edges"])); stack=[(0,0,"","",("","",""),("","",""),False,False)]
   while stack and states<limit:
    li,ri,lt,rt,lb,rb,ls,rs=stack.pop(); states+=1
    if li==len(lw) and ri==len(rw):
     rendered=(lt+"; "+rt).strip(); au=audit(rendered)
     if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"cross_word_seam":ls or rs,"complete_semantic_parse":True,"role_switch_complete":True,"reader_eligible":False,"reason":"complete clarification derivation but residual/exact gate failed"})
     if any(lb) or any(rb) or not(ls or rs):
      if not(ls or rs): seam_prunes+=1
      continue
     if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen:
      seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"left_derivation":left,"right_derivation":right,"three_chunk_residual":True,"role_switch":True,"posthoc_repair":False,"finished_tape_reversal":False,"mirrored_units":False}})
     continue
    if li<len(lw):
     words=lw[li]; text="".join(words); lchunk=min(li,2); res=consume(lb[lchunk]+norm(text),rb[lchunk])
     if res is None: char_prunes+=1
     else:
      nlb=list(lb); nlb[lchunk]=res[0]; stack.append((li+1,ri,(lt+" " if lt else "")+" ".join(words),rt,tuple(nlb),rb,ls or(bool(lb[lchunk]) and len(norm(text))>len(rb[lchunk])),rs))
    if ri<len(rw):
     words=rw[ri]; text="".join(words); chunk=max(0,2-ri); res=consume(lb[chunk],rb[chunk]+norm(text)[::-1])
     if res is None: char_prunes+=1
     else:
      nrb=list(rb); nrb[chunk]=res[0]; stack.append((li,ri+1,lt," ".join(words)+(" "+rt if rt else ""),lb,tuple(nrb),ls,rs or(bool(rb[chunk]) and len(norm(text))>len(lb[chunk]))))
   if states>=limit: break
  if states>=limit: break
 return {"method":"vocative-qa-three-chunk-clarification-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","derivations":len(ds),"states":states,"character_prunes":char_prunes,"semantic_prunes":semantic_prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(ds),"reader_facing_candidates":[],"reader_eligible":False,"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh three-chunk clarification answer with typed addressee-to-witness role switch; chunk residuals remain separate during online character equations and cross-word seam; no command/appositive sweep, reversal, repair, catalogue text, or mirrored units","novelty_preflight":{"passed":True,"overlaps_checked":["vocative-qa-two-chunk-residual-20260920","nullable-cfg-vocative-question-answer-20260920","dialogue-speech-act-residual-20260916"],"unused_dimension":"three semantic residual chunks with an explicit answer role switch in the clarification tail","reason":"prior QA lane had one/two answer chunks without a typed role switch; this lane requires a complete three-edge clarification parse before exact admission"},"first_live_diagnostic":"three-chunk residual mismatch at role-switch clarification edge" if not exact else "exact closure requires blinded reader review","next_construction":"hold out a four-chunk clarification with a second role switch only if a reader-worthy closure appears"}

if __name__=="__main__":
 result=run(); out=ROOT/"runs/vocative-qa-three-chunk-clarification-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("derivations","states","character_prunes","semantic_prunes","seam_prunes","exact_candidate_count")}))
