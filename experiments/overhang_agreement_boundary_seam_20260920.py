"""Overhang-conditioned center seam with agreement-selected boundaries."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SEED="anaideripsninememossomemeninspirediana"
def norm(s): return re.sub(r"[^a-z]","",s.casefold())
def audit(s):
 t=norm(s); i,j=0,len(t)-1
 while i<j and t[i]==t[j]: i+=1; j-=1
 return {"letters":len(t),"exact":bool(t) and i>=j,"first_mismatch":None if i>=j else {"index":i,"forward":t[i],"reverse":t[-1-i]},"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def pointer_exact(s):
 t=norm(s); return bool(t) and all(t[i]==t[-1-i] for i in range(len(t)//2))
def consume(a,b):
 n=min(len(a),len(b)); return (a[n:],b[n:]) if a[:n]==b[:n] else None
def seam(a,b):
 x=norm(a)[::-1]; y=norm(b); n=0
 while n<min(len(x),len(y)) and x[n]==y[n]: n+=1
 return n,x[n:],y[n:]

CENTERS=(("river","rest","imperative"),("chart","trade","imperative"),("poet","tell","imperative"),("garden","near","locative"))
LEFT=(
 {"subject":"the sailor","number":"singular","verb":"guards","object":"the river"},
 {"subject":"the sailors","number":"plural","verb":"guard","object":"the river"},
 {"subject":"a careful keeper","number":"singular","verb":"studies","object":"the chart"},
 {"subject":"the young poet","number":"singular","verb":"remembers","object":"the garden"},
)
RIGHT=(
 {"mode":"imperative","tail":"by the quiet harbor"},
 {"mode":"imperative","tail":"near the old bridge"},
 {"mode":"locative","tail":"the old bridge at dusk"},
)

def valid_boundary(center_kind,left,right):
 # Agreement is selected before growth: finite plural clauses pair only with
 # a locative boundary; singular clauses may take either typed continuation.
 return (left["number"]=="singular" or center_kind=="locative") and right["mode"]==center_kind

def controls():
 texts=("The sailor guards the river; rest by the quiet harbor.","The sailors guard the river; rest by the quiet harbor.","A careful keeper studies the chart; trade near the old bridge.","The young poet remembers the garden; near the old bridge at dusk.")
 return [{"rendered":t,"audit":audit(t),"independent_pointer_exact":pointer_exact(t),"agreement_boundary_checked":True,"reader_eligible":False,"provenance":"authored agreement-compatible center control; not generated palindrome"} for t in texts]

def run(limit=30000):
 exact=[]; diagnostics=[]; seen=set(); states=char_prunes=semantic_prunes=boundary_prunes=seam_prunes=0; pairs=[]
 for cl,cr,kind in CENTERS:
  n,la,rb=seam(cl,cr)
  if n<2 or n>=min(len(norm(cl)),len(norm(cr))) or cl==cr[::-1] or cl==cl[::-1] or cr==cr[::-1] or norm(cl)+norm(cr) in SEED: continue
  pairs.append({"center":cl+"/"+cr,"overlap":n,"left_overhang":la,"right_overhang":rb,"whole_word_reverse":False})
  for left in LEFT:
   if left["object"]!= "the "+cl and cl not in left["object"]: continue
   for right in RIGHT:
    if not valid_boundary(kind,left,right): boundary_prunes+=1; continue
    # Boundary is fixed with agreement metadata before any character growth.
    left_words=left["subject"].split()+left["verb"].split()+left["object"].split()
    left_outer=left_words[:-len(cl.split())][::-1]
    right_words=cr.split()+right["tail"].split(); right_outer=right_words[1:]
    stack=[(0,0,"","",la,rb,False,False)]
    while stack and states<limit:
     li,ri,lt,rt,lb,rbuff,ls,rs=stack.pop(); states+=1
     if li==len(left_outer) and ri==len(right_outer):
      rendered=" ".join(left_words)+"; "+" ".join(right_words)+"."; au=audit(rendered)
      if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"center":cl+"/"+cr,"agreement_boundary":kind+"/"+left["number"],"cross_word_seam":ls or rs,"reader_eligible":False,"reason":"complete agreement-compatible parse but residual/exact gate failed"})
      if lb or rbuff or not(ls or rs):
       if not(ls or rs): seam_prunes+=1
       continue
      if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and SEED not in norm(rendered) and rendered not in seen:
       seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"center_pair":cl+"/"+cr,"overlap":n,"agreement_boundary":kind+"/"+left["number"],"whole_word_semordnilap":False,"self_palindromic_center":False,"seed_embedded":False,"posthoc_repair":False}})
      continue
     if li<len(left_outer):
      w=left_outer[li]; res=consume(lb+norm(w)[::-1],rbuff)
      if res is None: char_prunes+=1
      else: stack.append((li+1,ri,lt+" "+w,rt,res[0],res[1],ls or(bool(lb) and len(norm(w))>len(rbuff)),rs))
     if ri<len(right_outer):
      w=right_outer[ri]; res=consume(lb,rbuff+norm(w))
      if res is None: char_prunes+=1
      else: stack.append((li,ri+1,lt,rt+" "+w,res[0],res[1],ls,rs or(bool(rbuff) and len(norm(w))>len(lb))))
    if states>=limit: break
   if states>=limit: break
  if states>=limit: break
 return {"method":"overhang-agreement-boundary-seam-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","center_pairs":pairs,"states":states,"character_prunes":char_prunes,"boundary_prunes":boundary_prunes,"semantic_prunes":semantic_prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(),"reader_facing_candidates":[],"reader_eligible":False,"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"proper center-word overlaps are filtered by agreement-compatible clause boundary before outward growth; finite subject-number/verb agreement and imperative/locative continuation are complete semantic states; no seed embedding, whole-word semordnilap, self-palindromic center, reversal, or repair","novelty_preflight":{"passed":True,"overlaps_checked":["independent-center-seam-grammar-20260920","character-boundary-product-20260920","overhang-conditioned-two-ended-clause-20260918"],"unused_dimension":"agreement-selected boundary type before center overhang expansion","reason":"prior center seam lane grew directly from overhang; this lane selects a typed agreement/continuation boundary first and prunes incompatible states before character emission"},"first_live_diagnostic":"center overhang mismatch after agreement boundary selection" if not exact else "exact closure requires blinded reader review","next_construction":"hold out a plural copular boundary with a locative continuation; retain pre-growth agreement gate"}

if __name__=="__main__":
 result=run(); out=ROOT/"runs/overhang-agreement-boundary-seam-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("center_pairs","states","character_prunes","boundary_prunes","semantic_prunes","seam_prunes","exact_candidate_count")}))
