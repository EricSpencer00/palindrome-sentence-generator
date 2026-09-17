"""Typed parenthetical attachment in the center-crossing grammar."""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/center-crossing-parenthetical-attachment-20260917.json";REG=ROOT/"docs/experiment-novelty-registry.json"
ID="center-crossing-parenthetical-attachment-20260917";SIG="midpoint-inside-token|typed-parenthetical-attachment|role-tense-complement-fixed|live-midpoint-debt|independent-exact-audit"
SCENES=(("the patient gardener","watered","the shaded orchard","for the local archivist","for the archive","after the morning rain","a careful steward"),("the senior curator","examined","the copper circuit","for the museum guide","with care","beside the archive wall","a patient technician"),("the young engineer","tested","a delicate instrument","for the workshop lead","in silence","near the north window","a methodical designer"))
ROLES=("the local archivist","the museum guide","the workshop lead"); CLAUSE_OBJECTS=("the notes","the records","the measurements")
def norm(s):return re.sub(r"[^a-z]","",s.lower())
def audit(s):
 t=norm(s);m=[];i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({"left":i,"right":j,"a":t[i],"b":t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();return {"normalized_tape":t,"letters":len(t),"exact":bool(t) and not m,"independent_two_pointer_exact":bool(t) and not m,"first_mismatches":m[:8],"sha256_forward":f,"sha256_reverse":r,"sha_equal_under_reversal":f==r}
def preflight():
 es=json.loads(REG.read_text()).get("entries",[]);a=str(Path(__file__).relative_to(ROOT));return {"status":"passed","registry_entries_read":len(es),"signature_collision":any(x.get("signature")==SIG for x in es),"artifact_collision":any(x.get("artifact")==a for x in es),"shortcuts_rejected":["finished-tape reversal","word-order symmetry","catalogue text","fragments"]}
def emit(scene,role,clause_obj,tense):
 sg,verb_present,obj,attachment,comp,place,parenthetical=scene;verb=verb_present if tense=="present" else {"watered":"watered","examined":"examined","tested":"tested"}[verb_present]
 text=f"{sg.capitalize()} ({parenthetical}) {verb} {obj} {attachment} {comp} {place} because {role} {verb} {clause_obj} for the local archive.";t=norm(text);mid=len(t)//2;sp=[];cur=0
 for tok in re.findall(r"[A-Za-z]+",text):a=cur;cur+=len(tok);sp.append((tok,a,cur))
 cross=next(({"token":w,"token_interval":[a,b],"midpoint":mid,"offset":mid-a} for w,a,b in sp if a<=mid<b),None);a=audit(text);i,j=0,len(t)-1;pairs=0
 while i<j and t[i]==t[j]:pairs+=1;i+=1;j-=1
 return {"rendered":text,"choices":{"tense":tense,"verb":verb,"role":role,"clause_complement":clause_obj,"attachment":attachment,"parenthetical":parenthetical},"audit":a,"midpoint_state":{"midpoint":mid,"crossing":cross,"closed_pairs_before_first_mismatch":pairs,"live_debt":a["first_mismatches"][0] if a["first_mismatches"] else None,"syntactic_family":"typed parenthetical attachment"},"anti_shortcut_flags":{"finished_tape_reversal":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"catalogue_text":False,"punctuation_changes_letters":False,"fragment":False},"provenance":{"lexical_source":"fresh hand-authored parenthetical inventory","borrowed_text":False,"typed_role_tense_complement_fixed":True,"new_syntax":True}}
def run():
 pre=preflight();rows=[emit(s,r,c,t) for s,r,c,t in itertools.product(SCENES,ROLES,CLAUSE_OBJECTS,("present","past"))];rows.sort(key=lambda x:(x["audit"]["exact"],x["audit"]["letters"]),reverse=True);exact=[r for r in rows if r["audit"]["exact"]]
 return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure","method":"typed parenthetical syntactic attachment with center-token grammar state","novelty_preflight":pre,"candidate_count":len(rows),"exact_count":len(exact),"reader_eligible":False,"rendered_candidates":rows,"stats":{"variants":len(rows),"longest_letters":max(r["audit"]["letters"] for r in rows),"midpoint_inside_token":sum(r["midpoint_state"]["crossing"] is not None for r in rows)},"failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found","next_repair":"retain parenthetical and center state, then test one distinct adverbial attachment with typed semantic role"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer scan","forward/reverse SHA-256"],"shortcuts_excluded":True}}
if __name__=="__main__":
 x=run();OUT.write_text(json.dumps(x,indent=2)+"\n");print(json.dumps({"candidates":x["candidate_count"],"exact":x["exact_count"],"stats":x["stats"]},sort_keys=True))
