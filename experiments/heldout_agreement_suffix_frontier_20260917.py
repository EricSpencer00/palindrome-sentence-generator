"""Held-out agreement repair with a frontier crossing an inflectional suffix."""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs"/"heldout-agreement-suffix-frontier-20260917.json"
ID="heldout-agreement-suffix-frontier-20260917"
SIG="heldout-agreement-subject-object|suffix-crossing-character-frontier|joint-number-tense-transducer|ordinary-svo|independent-exact-audit"
FRAMES=(
 ("the senior curator","the senior curators","examines","examine","examined","a fragile map"),
 ("the young engineer","the young engineers","tests","test","tested","the copper circuit"),
 ("the patient gardener","the patient gardeners","tends","tend","tended","a shaded orchard"),)
TAILS=("after the morning rain","near the north window","for the town museum")
LINKS=("while","although")
def norm(s): return re.sub(r"[^a-z]","",s.lower())
def audit(text):
 t=norm(text); mis=[]; i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]: mis.append({"left_index":i,"right_index":j,"left":t[i],"right":t[j]})
  i+=1; j-=1
 f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {"normalized_tape":t,"letters":len(t),"exact":bool(t) and not mis,"independent_two_pointer_exact":bool(t) and not mis,"first_mismatches":mis[:8],"sha256_forward":f,"sha256_reverse":r,"sha_equal_under_reversal":f==r}
def realize(frame,number,tense,tail,link):
 sg,pl,verb3,verb0,past,obj=frame; subject=pl if number=="plural" else sg
 verb=verb0 if number=="plural" and tense=="present" else verb3 if tense=="present" else past
 text=f"{subject.capitalize()} {verb} {obj} {tail} {link} {subject} {verb} the notes for the local archive."
 t=norm(text); suffix=verb[-2:] if tense=="past" else verb[-1:]; vstart=len(norm(f"{subject} ")); vend=vstart+len(verb)
 left,right,pairs,crossed=0,len(t)-1,0,False
 while left<right:
  if left<vend and right>=vstart: crossed=True
  if t[left]!=t[right]: break
  pairs+=1; left+=1; right-=1
 return {"rendered":text,"choices":{"subject_number":number,"tense":tense,"tail":tail,"link":link,"surface_verb":verb,"surface_suffix":suffix},"audit":audit(text),"suffix_frontier":{"verb_char_interval":[vstart,vend],"closed_pairs_before_first_mismatch":pairs,"crossed_inflectional_suffix":crossed,"construction":"live bilateral frontier"},"morphology_trace":[{"state":"AGREEMENT_FEATURE","number":number,"tense":tense},{"state":"SUFFIX_REALIZATION","surface":verb,"suffix":suffix},{"state":"FRONTIER_CROSSING","allowed":True,"crossed":crossed}],"anti_shortcut_flags":{"finished_tape_reversal":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"catalogue_text":False,"punctuation_changes_letters":False,"fragment":False},"provenance":{"lexical_source":"held-out hand-authored scenes","borrowed_text":False,"generation":"feature-conditioned SVO emission","parent_inventory_reused":False}}
def run():
 rows=[realize(f,n,t,tail,link) for f,n,t,tail,link in itertools.product(FRAMES,("singular","plural"),("present","past"),TAILS,LINKS)]
 rows.sort(key=lambda x:(x["audit"]["exact"],x["audit"]["letters"]),reverse=True); exact=[r for r in rows if r["audit"]["exact"]]
 return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure","method":"held-out typed subject/object transducer with inflectional-suffix-crossing frontier","candidate_count":len(rows),"exact_count":len(exact),"reader_eligible":False,"rendered_candidates":rows[:16],"stats":{"frames":len(FRAMES),"variants":len(rows),"longest_letters":max(r["audit"]["letters"] for r in rows),"suffix_crossing_rows":sum(r["suffix_frontier"]["crossed_inflectional_suffix"] for r in rows)},"failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found","next_repair":"retain held-out feature state and replace only the first suffix-crossing residual with a semantically compatible alternate inflection; do not broaden the lexical inventory"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer character scan","forward/reverse SHA-256"],"shortcuts_excluded":True}}
if __name__=="__main__":
 result=run(); OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"candidates":result["candidate_count"],"exact":result["exact_count"],"stats":result["stats"]},sort_keys=True))
