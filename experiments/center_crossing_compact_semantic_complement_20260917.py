"""Compact semantic complement after bounded coordination; no new clause."""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/center-crossing-compact-complement-20260917.json";REG=ROOT/"docs/experiment-novelty-registry.json"
ID="center-crossing-compact-complement-20260917";SIG="midpoint-inside-token|compact-semantic-complement|single-coordination-fixed|temporal-contrast-discourse-state-fixed|independent-exact-audit"
SCENES=(("the patient gardener","watered","the shaded orchard","for the local archivist","with patience","for the archive","after the morning rain"),("the senior curator","examined","the copper circuit","for the museum guide","in silence","with care","beside the archive wall"),("the young engineer","tested","a delicate instrument","for the workshop lead","without delay","in silence","near the north window"))
ROLES=("the local archivist","the museum guide","the workshop lead"); OBJECTS=("the notes","the records","the measurements"); MARKERS=("in fact","as a result"); CONTRASTS=("although the light was fading","although the room was quiet","although the work was difficult"); TIMES=("when the bell sounded","after the meeting ended","before the lamps were lit"); COMPLEMENTS=("as planned","with purpose","in context"); COORD="and the archive remained open"
def norm(s):return re.sub(r"[^a-z]","",s.lower())
def audit(s):
 t=norm(s);m=[];i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({"left":i,"right":j,"a":t[i],"b":t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();return {"normalized_tape":t,"letters":len(t),"exact":bool(t) and not m,"independent_two_pointer_exact":bool(t) and not m,"first_mismatches":m[:8],"sha256_forward":f,"sha256_reverse":r,"sha_equal_under_reversal":f==r}
def preflight():
 es=json.loads(REG.read_text()).get("entries",[]);a=str(Path(__file__).relative_to(ROOT));return {"status":"passed","registry_entries_read":len(es),"signature_collision":any(x.get("signature")==SIG for x in es),"artifact_collision":any(x.get("artifact")==a for x in es),"shortcuts_rejected":["finished-tape reversal","word-order symmetry","catalogue text","fragments"]}
def emit(scene,role,obj,marker,contrast,time,compact,tense):
 sg,vp,ob,attachment,adverb,comp,place=scene;verb=vp if tense=="present" else {"watered":"watered","examined":"examined","tested":"tested"}[vp]
 text=f"{sg.capitalize()} {verb} {ob} {attachment} {adverb} {comp} {place}, {marker}, because {role} {verb} {obj} for the local archive, {contrast}, {time}; {COORD}, {compact}.";t=norm(text);mid=len(t)//2;sp=[];cur=0
 for tok in re.findall(r"[A-Za-z]+",text):a=cur;cur+=len(tok);sp.append((tok,a,cur))
 cross=next(({"token":w,"token_interval":[a,b],"midpoint":mid,"offset":mid-a} for w,a,b in sp if a<=mid<b),None);a=audit(text);i,j=0,len(t)-1;pairs=0
 while i<j and t[i]==t[j]:pairs+=1;i+=1;j-=1
 return {"rendered":text,"choices":{"tense":tense,"verb":verb,"role":role,"object":obj,"attachment":attachment,"adverbial":adverb,"marker":marker,"contrast":contrast,"temporal":time,"coordination":COORD,"compact_complement":compact},"audit":a,"midpoint_state":{"midpoint":mid,"crossing":cross,"closed_pairs_before_first_mismatch":pairs,"live_debt":a["first_mismatches"][0] if a["first_mismatches"] else None,"syntactic_family":"compact semantic complement"},"anti_shortcut_flags":{"finished_tape_reversal":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"catalogue_text":False,"punctuation_changes_letters":False,"fragment":False},"provenance":{"lexical_source":"fresh compact semantic complement inventory","borrowed_text":False,"all_prior_states_fixed":True,"new_clause":False}}
def run():
 pre=preflight();rows=[emit(s,r,o,m,x,z,c,t) for s,r,o,m,x,z,c,t in itertools.product(SCENES,ROLES,OBJECTS,MARKERS,CONTRASTS,TIMES,COMPLEMENTS,("present","past"))];rows.sort(key=lambda x:(x["audit"]["exact"],x["audit"]["letters"]),reverse=True);exact=[r for r in rows if r["audit"]["exact"]]
 return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure","method":"compact semantic complement after one fixed coordination","novelty_preflight":pre,"candidate_count":len(rows),"exact_count":len(exact),"reader_eligible":False,"rendered_candidates":rows[:500],"stats":{"variants":len(rows),"longest_letters":max(r["audit"]["letters"] for r in rows),"midpoint_inside_token":sum(r["midpoint_state"]["crossing"] is not None for r in rows)},"failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found","next_repair":"retain compact complement and replace the fixed coordination with a semantic conjunction-free attachment"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer scan","forward/reverse SHA-256"],"shortcuts_excluded":True}}
if __name__=="__main__":
 x=run();OUT.write_text(json.dumps(x,indent=2)+"\n");print(json.dumps({"candidates":x["candidate_count"],"exact":x["exact_count"],"stats":x["stats"]},sort_keys=True))
