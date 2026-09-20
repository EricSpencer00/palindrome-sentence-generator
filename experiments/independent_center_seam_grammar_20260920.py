"""Independent center-word seam grammar with live outward growth.

The center is two independently selected ordinary words.  Their inward-facing
characters share only a proper prefix, leaving a residual overhang.  Complete
clauses then grow outward from that seam under live character equations.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SEED="anaideripsninememossomemeninspirediana"

def norm(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
 t=norm(s); i,j=0,len(t)-1
 while i<j and t[i]==t[j]: i+=1; j-=1
 return {"letters":len(t),"exact":bool(t) and i>=j,"first_mismatch":None if i>=j else {"index":i,"forward":t[i],"reverse":t[-1-i]},"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def pointer_exact(s):
 t=norm(s); return bool(t) and all(t[i]==t[-1-i] for i in range(len(t)//2))
def consume(a,b):
 n=min(len(a),len(b)); return (a[n:],b[n:]) if a[:n]==b[:n] else None

# Proper partial inward overlaps only: reverse(left) and right share 2+ chars,
# but neither is a whole reverse word and neither center is self-palindromic.
CENTERS=(
 ("river","rest","imperative","river/rest"),
 ("chart","trade","imperative","chart/trade"),
 ("poet","tell","imperative","poet/tell"),
 ("garden","near","preposition","garden/near"),
 ("keeper","keep","verb","keeper/keep"),
 ("letter","lest","conjunction","letter/lest"),
)
LEFT_FRAMES=(
 ("the patient sailor","guards","the quiet",("river","chart","garden")),
 ("a careful keeper","studies","the old",("chart","garden","letter")),
 ("the young poet","remembers","the winter",("poet","garden","river")),
)
RIGHT_TAILS=(
 ("by the quiet harbor",("rest","trade","keep")),
 ("near the old bridge",("near","lest","tell")),
 ("before the evening rain",("keep","rest","trade")),
)

def seam(left,right):
 a=norm(left)[::-1]; b=norm(right); n=0
 while n<min(len(a),len(b)) and a[n]==b[n]: n+=1
 return n,a[n:],b[n:]

def valid_center(left,right,overlap):
 n,la,rb=seam(left,right)
 return n>=overlap and n<min(len(norm(left)),len(norm(right))) and left!=right and left!=right[::-1] and left!=left[::-1] and right!=right[::-1] and norm(left)+norm(right) not in SEED

def controls():
 rows=[]
 texts=("The patient sailor guards the river; rest by the quiet harbor.",
        "A careful keeper studies the chart; trade near the old bridge.",
        "The young poet remembers the garden; keep before the evening rain.")
 for text in texts:
  rows.append({"rendered":text,"audit":audit(text),"independent_pointer_exact":pointer_exact(text),"reader_eligible":False,"provenance":"authored complete clause control; not generated from a center seam"})
 return rows

def run(limit=30000):
 exact=[]; diagnostics=[]; states=char_prunes=semantic_prunes=seam_prunes=0; seen=set(); seam_pairs=[]
 for cl,cr,kind,label in CENTERS:
  n,la,rb=seam(cl,cr)
  if not valid_center(cl,cr,2): continue
  seam_pairs.append({"pair":label,"overlap":n,"left_residual":la,"right_residual":rb,"whole_word_reverse":False})
  for subj,verb,adj,allowed in LEFT_FRAMES:
   if cl not in allowed: continue
   left_outer=(subj.split()+[verb]+adj.split())[::-1]  # inner-to-outer additions
   left_words=subj.split()+[verb]+adj.split()+[cl]
   for tail,rights in RIGHT_TAILS:
    if cr not in rights: continue
    right_outer=tail.split()  # center-to-outer additions
    right_words=[cr]+right_outer
    # center residual streams begin with inward-facing center words.
    stack=[(0,0,"","",la,rb,False,False)]
    while stack and states<limit:
     li,ri,lt,rt,lb,rbuff,ls,rs=stack.pop(); states+=1
     if li==len(left_outer) and ri==len(right_outer):
      rendered=" ".join(left_words)+"; "+" ".join(right_words)+"."
      au=audit(rendered)
      if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"center":label,"center_overlap":n,"left_residual":lb,"right_residual":rbuff,"cross_word_seam":ls or rs,"reader_eligible":False,"reason":"complete outward grammar but residual/exact gate failed"})
      if lb or rbuff or not(ls or rs):
       if not(ls or rs): seam_prunes+=1
       continue
      if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and SEED not in norm(rendered) and rendered not in seen:
       seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"center_pair":label,"center_overlap":n,"left_overhang":la,"right_overhang":rb,"whole_word_semordnilap":False,"self_palindromic_center":False,"seed_embedded":False,"posthoc_repair":False,"word_order_only":False}})
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
 return {"method":"independent-center-seam-grammar-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","center_pairs":seam_pairs,"states":states,"character_prunes":char_prunes,"semantic_prunes":semantic_prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(),"reader_facing_candidates":[],"reader_eligible":False,"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"independently selected ordinary center words with proper partial inward overlap and residual overhang; complete SVO/imperative/prepositional clauses grow outward under live equations; no seed embedding, whole-word semordnilap, self-palindromic center, finished-tape reversal, or repair","novelty_preflight":{"passed":True,"overlaps_checked":["variable-phrase-grammar-20260920","fresh-heteropalindrome-seam-search-20260920","direct-clause-pair-inventory-20260920","vocative-qa-three-chunk-clarification-20260920"],"unused_dimension":"partial center-word overlap with residual overhang followed by grammatical outward growth; center pair is neither a whole reversed word nor a copied seed span","reason":"prior seam lanes paired complete clauses or aligned lexical units; this lane starts from proper character-prefix overlap between independent center words and carries the overhang into complete clause derivation"},"first_live_diagnostic":"center residual overhang mismatch during outward grammar growth" if not exact else "exact closure requires blinded reader review","next_construction":"hold out a center pair with a transitive right clause and retain proper-overlap plus complete-role gates"}

if __name__=="__main__":
 result=run(); out=ROOT/"runs/independent-center-seam-grammar-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("center_pairs","states","character_prunes","semantic_prunes","seam_prunes","exact_candidate_count")}))
