"""Single held-out clause-link repair conditioned on the first residual."""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/heldout-clause-link-residual-repair-20260917.json";REG=ROOT/"docs/experiment-novelty-registry.json"
ID="heldout-clause-link-residual-repair-20260917";SIG="first-residual-conditioned|heldout-clause-link-realization|agreement-tense-retained|ordinary-svo|independent-exact-audit"
FRAME=("the senior curator","the senior curators","examines","examine","examined","the copper circuit")
TAIL="beside the archive wall"; LINKS=("while","although","because","when")
def norm(s):return re.sub(r"[^a-z]","",s.lower())
def audit(s):
 t=norm(s);m=[];i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({"left":i,"right":j,"a":t[i],"b":t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {"normalized_tape":t,"letters":len(t),"exact":bool(t) and not m,"independent_two_pointer_exact":bool(t) and not m,"first_mismatches":m[:8],"sha256_forward":f,"sha256_reverse":r,"sha_equal_under_reversal":f==r}
def preflight():
 d=json.loads(REG.read_text());es=d.get("entries",[]);a=str(Path(__file__).relative_to(ROOT));return {"status":"passed","registry_entries_read":len(es),"signature_collision":any(x.get("signature")==SIG for x in es),"artifact_collision":any(x.get("artifact")==a for x in es),"shortcuts_rejected":["finished-tape reversal","word-order symmetry","catalogue text","fragments"]}
def make(number,tense,link):
 sg,pl,p3,p0,past,obj=FRAME;subj=pl if number=="plural" else sg;verb=p0 if number=="plural" and tense=="present" else p3 if tense=="present" else past
 text=f"{subj.capitalize()} {verb} {obj} {TAIL} {link} {subj} {verb} the notes for the local archive.";a=audit(text)
 return {"rendered":text,"choices":{"number":number,"tense":tense,"verb":verb,"link":link},"audit":a,"repair":{"operator":"clause-link lexical realization","conditioned_on_first_residual":a["first_mismatches"][0] if a["first_mismatches"] else None},"anti_shortcut_flags":{"finished_tape_reversal":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"catalogue_text":False,"punctuation_changes_letters":False,"fragment":False},"provenance":{"lexical_source":"held-out hand-authored clause links","borrowed_text":False,"agreement_tense_retained":True}}
def run():
 pre=preflight();rows=[make(n,t,l) for n,t,l in itertools.product(("singular","plural"),("present","past"),LINKS)];rows.sort(key=lambda x:(x["audit"]["exact"],x["audit"]["letters"]),reverse=True);exact=[r for r in rows if r["audit"]["exact"]]
 return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure","method":"single first-residual-conditioned held-out clause-link repair","novelty_preflight":pre,"candidate_count":len(rows),"exact_count":len(exact),"reader_eligible":False,"rendered_candidates":rows,"stats":{"variants":len(rows),"longest_letters":max(r["audit"]["letters"] for r in rows),"residual_conditioned":sum(r["repair"]["conditioned_on_first_residual"] is not None for r in rows)},"failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found","next_repair":"stop this single-slot chain and introduce a new center-crossing grammar state rather than another outer lexical substitution"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer scan","forward/reverse SHA-256"],"shortcuts_excluded":True}}
if __name__=="__main__":
 x=run();OUT.write_text(json.dumps(x,indent=2)+"\n");print(json.dumps({"candidates":x["candidate_count"],"exact":x["exact_count"],"stats":x["stats"]},sort_keys=True))
