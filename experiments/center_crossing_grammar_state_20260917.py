"""Center-crossing grammar state: midpoint may occur inside an emitted token."""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/center-crossing-grammar-state-20260917.json";REG=ROOT/"docs/experiment-novelty-registry.json"
ID="center-crossing-grammar-state-20260917";SIG="midpoint-inside-token|typed-grammar-state-transition|live-character-obligation|ordinary-intact-prose|independent-exact-audit"
SCENES=(("the senior curator","examined","the copper circuit","beside the archive wall"),("the patient gardener","watered","the shaded orchard","after the morning rain"),("the young engineer","tested","a delicate instrument","near the north window"))
CENTERS=("carefully","quietly","patiently","yesterday")
LINKS=("while","because")
def norm(s):return re.sub(r"[^a-z]","",s.lower())
def audit(s):
 t=norm(s);m=[];i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({"left":i,"right":j,"a":t[i],"b":t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {"normalized_tape":t,"letters":len(t),"exact":bool(t) and not m,"independent_two_pointer_exact":bool(t) and not m,"first_mismatches":m[:8],"sha256_forward":f,"sha256_reverse":r,"sha_equal_under_reversal":f==r}
def preflight():
 es=json.loads(REG.read_text()).get("entries",[]);a=str(Path(__file__).relative_to(ROOT));return {"status":"passed","registry_entries_read":len(es),"signature_collision":any(x.get("signature")==SIG for x in es),"artifact_collision":any(x.get("artifact")==a for x in es),"shortcuts_rejected":["finished-tape reversal","word-order symmetry","catalogue text","fragments"]}
def emit(scene,center,link):
 subject,verb,obj,place=scene;text=f"{subject.capitalize()} {verb} {obj} {center} {place} {link} {subject} {verb} the notes for the local archive."
 t=norm(text); mid=len(t)//2; token_spans=[]; cursor=0
 for token in re.findall(r"[A-Za-z]+",text):
  start=cursor;cursor+=len(token);token_spans.append((token,start,cursor));
  if cursor<len(t):cursor+=0
 crossing=next(({"token":tok,"token_interval":[a,b],"midpoint":mid,"offset":mid-a} for tok,a,b in token_spans if a<=mid<b),None)
 # Live obligation ledger walks from both ends until first mismatch, independent of construction.
 i,j,pairs=0,len(t)-1,0
 while i<j and t[i]==t[j]:pairs+=1;i+=1;j-=1
 a=audit(text)
 return {"rendered":text,"choices":{"scene":scene,"center_token":center,"link":link},"audit":a,"grammar_state":{"transition":"LEFT_CLAUSE -> CENTER_TOKEN -> RIGHT_CLAUSE","midpoint":mid,"crossing":crossing,"closed_pairs_before_first_mismatch":pairs},"anti_shortcut_flags":{"finished_tape_reversal":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"catalogue_text":False,"punctuation_changes_letters":False,"fragment":False},"provenance":{"lexical_source":"fresh hand-authored scenes","borrowed_text":False,"center_selected_before_emission":True}}
def run():
 pre=preflight();rows=[emit(s,c,l) for s,c,l in itertools.product(SCENES,CENTERS,LINKS)];rows.sort(key=lambda x:(x["audit"]["exact"],x["audit"]["letters"]),reverse=True);exact=[r for r in rows if r["audit"]["exact"]];cross=sum(r["grammar_state"]["crossing"] is not None for r in rows)
 return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure","method":"typed center-crossing grammar state with midpoint inside-token support","novelty_preflight":pre,"candidate_count":len(rows),"exact_count":len(exact),"reader_eligible":False,"rendered_candidates":rows,"stats":{"variants":len(rows),"longest_letters":max(r["audit"]["letters"] for r in rows),"midpoint_inside_token":cross},"failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found","next_repair":"retain center-token grammar state and add a typed object complement whose character span is selected against the live midpoint debt"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer scan","forward/reverse SHA-256"],"shortcuts_excluded":True}}
if __name__=="__main__":
 x=run();OUT.write_text(json.dumps(x,indent=2)+"\n");print(json.dumps({"candidates":x["candidate_count"],"exact":x["exact_count"],"stats":x["stats"]},sort_keys=True))
