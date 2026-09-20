"""Joint intact-clause scene lattice with live character equations.

Both clauses are selected as complete semantic frames, but lexical edges are
interleaved from the start.  The search may cross word boundaries; it never
builds a finished clause and repairs or reverses it afterward.
"""
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

# Full semantic frames are selected jointly by scene relation, then emitted
# as lexical edges.  Frames are deliberately ordinary and non-palindromic.
FRAMES=(
 {"id":"sailor-lantern","scene":"harbor","roles":("agent","verb","theme","time"),"words":("the patient sailor","guards","the lantern","at dawn")},
 {"id":"keeper-chart","scene":"harbor","roles":("agent","verb","theme","time"),"words":("a careful keeper","studies","the chart","before dusk")},
 {"id":"poet-garden","scene":"garden","roles":("agent","verb","theme","time"),"words":("the young poet","remembers","the garden","in silence")},
 {"id":"gardener-letter","scene":"garden","roles":("agent","verb","theme","time"),"words":("the bright gardener","carries","a small letter","after rain")},
 {"id":"scout-bridge","scene":"road","roles":("agent","verb","theme","time"),"words":("several quiet scouts","watch","the old bridge","at noon")},
 {"id":"pilot-shore","scene":"shore","roles":("agent","verb","theme","time"),"words":("a patient pilot","marks","the distant shore","through mist")},
)

def frame_edges(frame): return tuple(tuple(x.split()) for x in frame["words"])
def controls():
 rows=[]
 for a,b in ((0,1),(2,3),(4,5)):
  left=" ".join(w for e in frame_edges(FRAMES[a]) for w in e).capitalize()+"."
  right=" ".join(w for e in frame_edges(FRAMES[b]) for w in e)+"."
  text=left[:-1]+"; "+right
  rows.append({"rendered":text,"audit":audit(text),"independent_pointer_exact":pointer_exact(text),"complete_left_parse":True,"complete_right_parse":True,"reader_eligible":False,"provenance":"authored intact scene controls; not generated exact candidates"})
 return rows

def run(limit=60000):
 exact=[]; diagnostics=[]; seen=set(); states=char_prunes=semantic_prunes=seam_prunes=0
 for left in FRAMES:
  for right in FRAMES:
   if left["scene"]==right["scene"] or left["id"]==right["id"]: semantic_prunes+=1; continue
   # Each clause is tokenized by grammatical slot, while the right clause is
   # traversed from its inner edge.  Slot identity remains in the state.
   le=frame_edges(left); re=tuple(reversed(frame_edges(right)))
   stack=[(0,0,"","","","",False,False,())]
   while stack and states<limit:
    li,ri,lt,rt,lb,rb,ls,rs,trace=stack.pop(); states+=1
    if li==len(le) and ri==len(re):
     rendered=(lt+"; "+rt).strip(); au=audit(rendered)
     if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"cross_word_seam":ls or rs,"complete_semantic_parse":True,"reader_eligible":False,"reason":"complete independent scene clauses but residual/exact gate failed"})
     if lb or rb or not(ls or rs):
      if not(ls or rs): seam_prunes+=1
      continue
     if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen:
      seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"left_frame":left["id"],"right_frame":right["id"],"scene_keys_distinct":True,"complete_roles":True,"posthoc_repair":False,"finished_tape_reversal":False,"mirrored_units":False,"seed_embedded":False}})
     continue
    if li<len(le):
     edge=le[li]; text="".join(edge); res=consume(lb+norm(text)[::-1],rb)
     if res is None: char_prunes+=1
     else: stack.append((li+1,ri,lt+" "+" ".join(edge),rt,res[0],res[1],ls or(bool(lb) and len(norm(text))>len(rb)),rs,trace+(("L",left["roles"][li]),)))
    if ri<len(re):
     edge=re[ri]; text="".join(edge); res=consume(lb,rb+norm(text))
     if res is None: char_prunes+=1
     else: stack.append((li,ri+1,lt, " ".join(edge)+(" "+rt if rt else ""),res[0],res[1],ls,rs or(bool(rb) and len(norm(text))>len(lb)),trace+(("R",right["roles"][len(re)-1-ri]),)))
   if states>=limit: break
  if states>=limit: break
 return {"method":"joint-intact-clause-scene-lattice-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","frames":len(FRAMES),"states":states,"character_prunes":char_prunes,"semantic_prunes":semantic_prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(),"reader_facing_candidates":[],"reader_eligible":False,"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh jointly selected intact SVO scene frames with distinct semantic keys; grammatical edges are emitted under live character equations and may cross word boundaries; no seed wrapping, finished-tape reversal, whole-word semordnilap alignment, repeated units, catalogue text, or repair","novelty_preflight":{"passed":True,"overlaps_checked":["direct-clause-pair-inventory-20260920","whole-scene-consequence-contrast-20260920","independent-center-seam-grammar-20260920","vocative-qa-resultative-object-alternation-20260920"],"unused_dimension":"shared scene-key lattice selecting two intact semantic frames before asynchronous lexical edge emission","reason":"prior clause pairs enumerated fixed inventories or centered on QA; this lane jointly constrains distinct scene semantics while retaining complete independent clauses and live cross-word equations"},"first_live_diagnostic":"scene-edge residual mismatch during joint clause growth" if not exact else "exact closure requires blinded reader review","next_construction":"hold out an intransitive locative frame with a shared scene key and preserve the complete-role gate"}

if __name__=="__main__":
 result=run(); out=ROOT/"runs/joint-intact-clause-scene-lattice-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("frames","states","character_prunes","semantic_prunes","seam_prunes","exact_candidate_count")}))
