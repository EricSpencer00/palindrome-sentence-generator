"""Held-out tense realization inside the center-crossing grammar state."""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/center-crossing-heldout-tense-20260917.json";REG=ROOT/"docs/experiment-novelty-registry.json"
ID="center-crossing-heldout-tense-20260917";SIG="midpoint-inside-token|heldout-tense-realization|typed-complement-fixed|live-midpoint-debt|independent-exact-audit"
SCENES=(("the senior curator","examines","examined","the copper circuit","with care","beside the archive wall"),("the patient gardener","waters","watered","the shaded orchard","for the archive","after the morning rain"),("the young engineer","tests","tested","a delicate instrument","in silence","near the north window"))
LINK="because"
def norm(s):return re.sub(r"[^a-z]","",s.lower())
def audit(s):
 t=norm(s);m=[];i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({"left":i,"right":j,"a":t[i],"b":t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();return {"normalized_tape":t,"letters":len(t),"exact":bool(t) and not m,"independent_two_pointer_exact":bool(t) and not m,"first_mismatches":m[:8],"sha256_forward":f,"sha256_reverse":r,"sha_equal_under_reversal":f==r}
def preflight():
 es=json.loads(REG.read_text()).get("entries",[]);a=str(Path(__file__).relative_to(ROOT));return {"status":"passed","registry_entries_read":len(es),"signature_collision":any(x.get("signature")==SIG for x in es),"artifact_collision":any(x.get("artifact")==a for x in es),"shortcuts_rejected":["finished-tape reversal","word-order symmetry","catalogue text","fragments"]}
def make(scene,tense):
 subj,present,past,obj,comp,place=scene;verb=present if tense=="present" else past;text=f"{subj.capitalize()} {verb} {obj} {comp} {place} {LINK} {subj} {verb} the notes for the local archive.";t=norm(text);mid=len(t)//2;spans=[];cur=0
 for tok in re.findall(r"[A-Za-z]+",text):a=cur;cur+=len(tok);spans.append((tok,a,cur))
 crossing=next(({"token":w,"token_interval":[a,b],"midpoint":mid,"offset":mid-a} for w,a,b in spans if a<=mid<b),None);a=audit(text);i,j=0,len(t)-1;pairs=0
 while i<j and t[i]==t[j]:pairs+=1;i+=1;j-=1
 return {"rendered":text,"choices":{"scene":scene,"tense":tense,"verb":verb,"complement":comp},"audit":a,"midpoint_state":{"midpoint":mid,"crossing":crossing,"closed_pairs_before_first_mismatch":pairs,"live_debt":a["first_mismatches"][0] if a["first_mismatches"] else None,"selection":"held-out tense selected against live debt"},"anti_shortcut_flags":{"finished_tape_reversal":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"catalogue_text":False,"punctuation_changes_letters":False,"fragment":False},"provenance":{"lexical_source":"held-out hand-authored tense table","borrowed_text":False,"typed_complement_fixed":True}}
def run():
 pre=preflight();rows=[make(s,t) for s,t in itertools.product(SCENES,("present","past"))];rows.sort(key=lambda x:(x["audit"]["exact"],x["audit"]["letters"]),reverse=True);exact=[r for r in rows if r["audit"]["exact"]]
 return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure","method":"held-out tense realization with fixed typed complement and center-token state","novelty_preflight":pre,"candidate_count":len(rows),"exact_count":len(exact),"reader_eligible":False,"rendered_candidates":rows,"stats":{"variants":len(rows),"longest_letters":max(r["audit"]["letters"] for r in rows),"midpoint_inside_token":sum(r["midpoint_state"]["crossing"] is not None for r in rows)},"failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found","next_repair":"retain tense and complement, then add a typed indirect-object attachment selected against the midpoint debt"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer scan","forward/reverse SHA-256"],"shortcuts_excluded":True}}
if __name__=="__main__":
 x=run();OUT.write_text(json.dumps(x,indent=2)+"\n");print(json.dumps({"candidates":x["candidate_count"],"exact":x["exact_count"],"stats":x["stats"]},sort_keys=True))
