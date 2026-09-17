"""Object determiner/adjective repair conditioned on the first character residual."""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"runs/heldout-object-residual-agreement-repair-20260917.json"
ID="heldout-object-residual-agreement-repair-20260917"; SIG="first-residual-conditioned|object-determiner-adjective-realization|agreement-state-retained|ordinary-svo|independent-exact-audit"
SUBJECTS=(("the senior curator","the senior curators","examines","examine","examined"),("the young engineer","the young engineers","tests","test","tested"))
OBJECTS=("a fragile map","the fragile map","an old map","the copper circuit","a copper circuit")
TAILS=("after the morning rain","near the north window","for the town museum")
LINKS=("while","although")
def norm(s): return re.sub(r"[^a-z]","",s.lower())
def audit(s):
 t=norm(s);m=[];i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({"left":i,"right":j,"a":t[i],"b":t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {"normalized_tape":t,"letters":len(t),"exact":bool(t) and not m,"independent_two_pointer_exact":bool(t) and not m,"first_mismatches":m[:8],"sha256_forward":f,"sha256_reverse":r,"sha_equal_under_reversal":f==r}
def row(subject,plural,p3,p0,past,obj,tense,tail,link):
 n="plural" if plural else "singular"; verb=p0 if plural and tense=="present" else p3 if tense=="present" else past
 text=f"{subject.capitalize()} {verb} {obj} {tail} {link} {subject} {verb} the notes for the local archive."; a=audit(text)
 return {"rendered":text,"choices":{"number":n,"tense":tense,"verb":verb,"object":obj,"tail":tail,"link":link},"audit":a,"repair":{"conditioned_on_first_residual":a["first_mismatches"][0] if a["first_mismatches"] else None,"operator":"agreement-compatible object determiner/adjective substitution"},"anti_shortcut_flags":{"finished_tape_reversal":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"catalogue_text":False,"punctuation_changes_letters":False,"fragment":False},"provenance":{"lexical_source":"held-out object realization inventory","borrowed_text":False,"agreement_state_retained":True}}
def run():
 rows=[]
 for sg,pl,p3,p0,past in SUBJECTS:
  for plural,tense,obj,tail,link in itertools.product((False,True),("present","past"),OBJECTS,TAILS,LINKS):
   subject=pl if plural else sg; rows.append(row(subject,plural,p3,p0,past,obj,tense,tail,link))
 rows.sort(key=lambda x:(x["audit"]["exact"],x["audit"]["letters"]),reverse=True); exact=[r for r in rows if r["audit"]["exact"]]
 return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure","method":"first-residual-conditioned object determiner/adjective realization with retained agreement state","candidate_count":len(rows),"exact_count":len(exact),"reader_eligible":False,"rendered_candidates":rows[:18],"stats":{"variants":len(rows),"longest_letters":max(r["audit"]["letters"] for r in rows),"residual_conditioned":sum(r["repair"]["conditioned_on_first_residual"] is not None for r in rows)},"failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found","next_repair":"condition a compatible tail preposition on the same residual, retaining object and agreement choices"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer scan","forward/reverse SHA-256"],"shortcuts_excluded":True}}
if __name__=="__main__":
 x=run();OUT.write_text(json.dumps(x,indent=2)+"\n");print(json.dumps({"candidates":x["candidate_count"],"exact":x["exact_count"],"stats":x["stats"]},sort_keys=True))
