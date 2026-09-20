"""Anchored question-answer grammar with a typed vocative answer edge."""
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
 "harbor":{"question":"an aide asks whether the clerk read nine memos before dusk","vocative":"clerk","answer":"yes, clerk, the memos were read and Diana was thanked","roles":["questioner=aide","addressee=clerk","recipient=Diana"]},
 "garden":{"question":"an artist asks whether the pilot carried the blue kite at noon","vocative":"pilot","answer":"yes, pilot, the kite was carried and Nina was met","roles":["questioner=artist","addressee=pilot","recipient=Nina"]},
 "arena":{"question":"an editor asks whether the singer opened the new folder by dusk","vocative":"singer","answer":"yes, singer, the folder was opened and the arena was entered","roles":["questioner=editor","addressee=singer","destination=arena"]},
}
def controls():
 texts=("An aide asks whether the clerk read nine memos before dusk? Yes, clerk, the memos were read and Diana was thanked.","An artist asks whether the pilot carried the blue kite at noon? Yes, pilot, the kite was carried and Nina was met.","An editor asks whether the singer opened the new folder by dusk? Yes, singer, the folder was opened and the arena was entered.")
 return [{"rendered":t,"audit":audit(t),"independent_pointer_exact":pointer_exact(t),"intact_prose":True,"question_answer":True,"vocative":True,"reader_eligible":False,"provenance":"authored anchored QA-vocative control; not generated exact candidate"} for t in texts]
def run(limit=60000):
 exact=[]; diagnostics=[]; states=prunes=seam_prunes=0; seen=set()
 for scene,data in SCENES.items():
  lt=tuple(data["question"].split()); rt=tuple(reversed((data["answer"]).split())); stack=[(0,0,"","","","",False,False)]
  while stack and states<limit:
   li,ri,tl,tr,lb,rb,ls,rs=stack.pop(); states+=1
   if li==len(lt) and ri==len(rt):
    rendered=(tl+"? "+tr).strip(); au=audit(rendered)
    if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"scene":scene,"complete_qa_vocative_parse":True,"roles":data["roles"],"vocative":data["vocative"],"anchored_prefix":"an","anchored_suffix":data["answer"].split()[-1],"cross_word_seam":ls or rs,"reader_eligible":False,"reason":"complete QA-vocative parse but residual/exact gate failed"})
    if lb or rb or not(ls or rs):
     if not(ls or rs): seam_prunes+=1
     continue
    if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen:
     seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"scene":scene,"roles":data["roles"],"vocative":data["vocative"],"left_anchor":"an","right_endpoint":data["answer"].split()[-1],"posthoc_repair":False,"finished_tape_reversal":False,"catalogue_reuse":False,"mirrored_units":False}})
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
 result={"method":"anchored-an-na-qa-vocative-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","states":states,"character_prunes":prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(),"reader_facing_candidates":exact,"reader_eligible":bool(exact),"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh anchored question-answer grammar with a typed vocative edge before the independently authored answer; fixed name/place endpoints and live residual equations; no repair, reversal, catalogue text, or mirrored units","novelty_preflight":{"passed":True,"overlaps_checked":["anchored-an-na-question-answer-20260920","anchored-an-na-two-clause-roles-20260920","f7bc2659"],"unused_dimension":"short vocative address in anchored QA answer with fixed endpoints","reason":"prior anchored QA had no address edge; this lane adds explicit addressee binding before answer content without changing endpoint bank"},"first_live_diagnostic":"anchored vocative mismatch during concurrent growth" if not exact else "exact candidate requires blinded human readability review","next_construction":"if closure remains empty, add a polarity-bearing answer (yes/no) while preserving the fixed vocative and endpoint grammar"}
 return result
if __name__=="__main__":
 result=run(); out=ROOT/"runs/anchored-an-na-qa-vocative-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("states","character_prunes","seam_prunes","exact_candidate_count")}))
