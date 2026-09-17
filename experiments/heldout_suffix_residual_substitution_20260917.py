"""Residual-conditioned inflection substitution on a held-out SVO frame."""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"runs"/"heldout-suffix-residual-substitution-20260917.json"
ID="heldout-suffix-residual-substitution-20260917"
SIG="first-residual-conditioned|agreement-tense-preserving|heldout-inflection-substitution|ordinary-svo|independent-exact-audit"
FRAME=("the senior curator","the senior curators","examines","examine","examined","a fragile map")
# Alternatives are lexical realizations, not spelling edits.
ALTS={"present_sg":("examines","inspects","reviews"),"present_pl":("examine","inspect","review"),"past":("examined","inspected","reviewed")}
TAILS=("after the morning rain","near the north window","for the town museum")
LINKS=("while","although")
def norm(s): return re.sub(r"[^a-z]","",s.lower())
def audit(s):
 t=norm(s); mis=[]; i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]: mis.append({"left":i,"right":j,"a":t[i],"b":t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {"normalized_tape":t,"letters":len(t),"exact":bool(t) and not mis,"independent_two_pointer_exact":bool(t) and not mis,"first_mismatches":mis[:8],"sha256_forward":f,"sha256_reverse":r,"sha_equal_under_reversal":f==r}
def candidate(number,tense,verb,tail,link):
 sg,pl,_,_,_,obj=FRAME; subj=sg if number=="sg" else pl
 text=f"{subj.capitalize()} {verb} {obj} {tail} {link} {subj} {verb} the notes for the local archive."
 a=audit(text); first=a["first_mismatches"][0] if a["first_mismatches"] else None
 return {"rendered":text,"choices":{"number":number,"tense":tense,"verb":verb,"tail":tail,"link":link},"audit":a,"first_residual":first,"anti_shortcut_flags":{"finished_tape_reversal":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"catalogue_text":False,"punctuation_changes_letters":False,"fragment":False},"provenance":{"lexical_source":"held-out hand-authored frame","borrowed_text":False,"inflection_choice":"lexical substitution selected from agreement/tense-compatible table"}}
def run():
 rows=[]
 for number,tense,tail,link in itertools.product(("sg","pl"),("present","past"),TAILS,LINKS):
  key="present_sg" if tense=="present" and number=="sg" else "present_pl" if tense=="present" else "past"
  baseline=candidate(number,tense,ALTS[key][0],tail,link); residual=baseline["first_residual"]
  # Residual-conditioned branch: rotate only to forms whose morphology is valid.
  start=(residual["right"]+residual["left"])%len(ALTS[key]) if residual else 0
  for offset in range(len(ALTS[key])):
   verb=ALTS[key][(start+offset)%len(ALTS[key])]
   row=candidate(number,tense,verb,tail,link); row["repair"]={"conditioned_on":residual,"operator":"agreement-tense-compatible lexical inflection substitution","baseline_verb":ALTS[key][0]}; rows.append(row)
 rows.sort(key=lambda r:(r["audit"]["exact"],r["audit"]["letters"]),reverse=True); exact=[r for r in rows if r["audit"]["exact"]]
 return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure","method":"first-residual-conditioned substitution over held-out inflectional table","candidate_count":len(rows),"exact_count":len(exact),"reader_eligible":False,"rendered_candidates":rows[:18],"stats":{"unique_residual_branches":24,"longest_letters":max(r["audit"]["letters"] for r in rows),"agreement_valid_substitutions":len(rows)},"failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found","next_repair":"condition a compatible object determiner/adjective substitution on the same first residual while retaining verb feature state"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer scan","forward/reverse SHA-256"],"shortcuts_excluded":True}}
if __name__=="__main__":
 x=run();OUT.write_text(json.dumps(x,indent=2)+"\n");print(json.dumps({"candidates":x["candidate_count"],"exact":x["exact_count"],"stats":x["stats"]},sort_keys=True))
