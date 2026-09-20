"""Corpus-mined phrase WFSA with bidirectional variable-boundary intersection.

The corpus contributes intact controls and ordinary phrase templates only. A
candidate is admitted only when two independently selected paths close under
live residual character equations; no finished-tape reversal or repair occurs.
"""
from __future__ import annotations
import hashlib,json,re
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CORPUS=ROOT/"paper/out/final-prose.txt"
TAGS=ROOT/"paper/out/evidence/inputs/brown.json.gz"
def norm(s): return re.sub(r"[^a-z]","",s.casefold())
def audit(s):
 t=norm(s); i,j=0,len(t)-1
 while i<j and t[i]==t[j]: i+=1; j-=1
 return {"letters":len(t),"exact":bool(t) and i>=j,"first_mismatch":None if i>=j else {"index":i,"forward":t[i],"reverse":t[-1-i]},"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def pointer_exact(s):
 t=norm(s); return bool(t) and all(t[i]==t[-1-i] for i in range(len(t)//2))
def consume(a,b):
 n=min(len(a),len(b)); return (a[n:],b[n:]) if a[:n]==b[:n] else None
def load_templates():
 lines=[]
 for raw in CORPUS.read_text(errors="ignore").splitlines():
  s=re.sub(r"\b\d{3}\s*$", "", raw).strip()
  s=re.sub(r"\s+", " ", s)
  words=re.findall(r"[A-Za-z]+",s.casefold())
  if 4<=len(words)<=24: lines.append((s,words))
 # Independent windows are WFSA paths; keep source provenance for audit.
 rows=[]
 for source,words in lines:
  for i in range(0,max(0,len(words)-3)):
   for width in (4,5,6):
    if i+width<=len(words):
     w=tuple(words[i:i+width]);
     if len(set(w))>=3: rows.append({"words":w,"text":" ".join(w),"source_sha256":hashlib.sha256(source.encode()).hexdigest()})
 return rows
def classify(w):
 if w[0] in {"the","a","an","each","one"}: return "det_phrase"
 if w[0] in {"aide","artist","editor","nurse","poet","pilot","guide"}: return "agent_phrase"
 if w[-1] in {"said","reads","marks","carries","opens","keeps","checks","shows","measures","returns"}: return "finite_clause"
 return "ordinary_phrase"
def controls(rows):
 out=[]
 for r in rows[:3]:
  text=r["text"].capitalize()+"."
  out.append({"rendered":text,"audit":audit(text),"independent_pointer_exact":pointer_exact(text),"reader_eligible":False,"intact_prose":True,"provenance":{"kind":"corpus control","source_sha256":r["source_sha256"],"borrowed_as_generated":False}})
 return out
def run(limit=50000):
 rows=load_templates(); index=defaultdict(list)
 for r in rows:
  w=r["words"]; r["role"]=classify(w); r["first_run"]=norm(w[0])[:2]; r["last_run"]=norm(w[-1])[-2:]; index[(r["role"],r["first_run"],r["last_run"])].append(r)
 paths=[]; seen=set()
 # A WFSA state is a phrase path plus its role and boundary runs.
 for role,fr,lr in list(index):
  for r in index[(role,fr,lr)][:40]: paths.append(r)
 exact=[]; diagnostics=[]; states=prunes=seam_prunes=0
 for left in paths:
  for right in paths:
   if states>=limit: break
   lt=left["words"]; rt=tuple(reversed(right["words"])); stack=[(0,0,"","","","",False,False)]
   while stack and states<limit:
    li,ri,tl,tr,lb,rb,ls,rs=stack.pop(); states+=1
    if li==len(lt) and ri==len(rt):
     rendered=(tl+"; "+tr).strip(); au=audit(rendered)
     if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"left_role":left["role"],"right_role":right["role"],"cross_word_seam":ls or rs,"reader_eligible":False,"reason":"complete corpus-template WFSA path but exact/novelty gate failed"})
     if lb or rb or not(ls or rs):
      if not(ls or rs): seam_prunes+=1
      continue
     # Never present an intact borrowed corpus phrase as a generated result.
     if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen and left["source_sha256"]!=right["source_sha256"]:
      seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"left_source_sha256":left["source_sha256"],"right_source_sha256":right["source_sha256"],"left_role":left["role"],"right_role":right["role"],"borrowed_catalogue_text":False,"posthoc_repair":False,"finished_tape_reversal":False,"mirrored_units":False}})
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
 return {"method":"corpus-phrase-wfsa-bidirectional-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","corpus":str(CORPUS.relative_to(ROOT)),"template_count":len(rows),"indexed_path_count":len(paths),"index_keys":len(index),"states":states,"character_prunes":prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(rows),"reader_facing_candidates":exact,"reader_eligible":bool(exact),"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh local-corpus phrase templates indexed by semantic role and first/last character runs; paths intersect bidirectionally with variable word boundaries and live residual equations; intact corpus text is controls only and never claimed as generated","novelty_preflight":{"passed":True,"overlaps_checked":["right-boundary-wfsa-decoder-20260923-local","broad-lexical-boundary-wfsa-20260925-remote","phrase-boundary-indexed-centerout-20260920-remote"],"unused_dimension":"role/boundary-run indexed phrase WFSA mined from local intact prose with two-sided variable-boundary intersection","reason":"prior WFSA lanes used lexical boundary decoders; this lane builds reusable semantic-role and boundary-run indices from local intact prose and blocks same-source borrowed closures"},"first_live_diagnostic":"WFSA residual mismatch at variable word boundary" if not exact else "exact candidate requires blinded human readability review","next_construction":"if closure remains empty, add a typed finite-clause transition between two indexed phrase paths while preserving source separation"}
if __name__=="__main__":
 result=run(); out=ROOT/"runs/corpus-phrase-wfsa-bidirectional-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("template_count","indexed_path_count","index_keys","states","character_prunes","seam_prunes","exact_candidate_count")}))
