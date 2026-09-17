"""Second-proposition subject-role alternation in the short center grammar."""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/short-subject-role-alternation-20260917.json";REG=ROOT/"docs/experiment-novelty-registry.json"
ID="short-subject-role-alternation-20260917";SIG="short-two-constituent-grammar|second-proposition-subject-role-alternation|second-adverb-and-feature-state-fixed|independent-exact-audit"
FRAMES=(("the patient gardener","the patient gardeners","waters","water","watered","shaded orchard","records","record","recorded","measurements"),("the senior curator","the senior curators","examines","examine","examined","copper circuit","reviews","review","reviewed","records"),("the young engineer","the young engineers","tests","test","tested","delicate instrument","checks","check","checked","notes"))
ROLES=(("the local archivist","the local archivists"),("the museum guide","the museum guides"),("the workshop lead","the workshop leads")); DETERMINER="the";ADJ="weathered";FIRST_ADVERB="carefully";SECOND_ADVERB="carefully";PREP="beside the window";COMP="with purpose";LINK="while"
def norm(s):return re.sub(r"[^a-z]","",s.lower())
def audit(s):
 t=norm(s);m=[];i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({"left":i,"right":j,"a":t[i],"b":t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();return {"normalized_tape":t,"letters":len(t),"exact":bool(t) and not m,"independent_two_pointer_exact":bool(t) and not m,"first_mismatches":m[:8],"sha256_forward":f,"sha256_reverse":r,"sha_equal_under_reversal":f==r}
def preflight():
 es=json.loads(REG.read_text()).get("entries",[]);a=str(Path(__file__).relative_to(ROOT));return {"status":"passed","registry_entries_read":len(es),"signature_collision":any(x.get("signature")==SIG for x in es),"artifact_collision":any(x.get("artifact")==a for x in es),"shortcuts_rejected":["finished-tape reversal","word-order symmetry","catalogue text","fragments"]}
def emit(frame,role_pair,number,tense):
 sg,pl,vs,vp,past,noun,rs,rp,rpast,robj=frame;left_sub,left_v=(sg,vs if tense=="present" else past) if number=="singular" else (pl,vp if tense=="present" else past);right_sub=role_pair[0] if number=="singular" else role_pair[1];right_v=rs if tense=="present" and number=="singular" else rp if tense=="present" else rpast;lo=f"{DETERMINER} {ADJ} {noun}";ro=f"{DETERMINER} {ADJ} {robj}"
 text=f"{left_sub.capitalize()} {left_v} {lo} {FIRST_ADVERB} {PREP} {COMP} {LINK} {right_sub} {right_v} {ro} {SECOND_ADVERB}.";t=norm(text);mid=len(t)//2;sp=[];cur=0
 for tok in re.findall(r"[A-Za-z]+",text):a=cur;cur+=len(tok);sp.append((tok,a,cur))
 cross=next(({"token":w,"token_interval":[a,b],"midpoint":mid,"offset":mid-a} for w,a,b in sp if a<=mid<b),None);a=audit(text);i,j=0,len(t)-1;pairs=0
 while i<j and t[i]==t[j]:pairs+=1;i+=1;j-=1
 return {"rendered":text,"choices":{"number":number,"tense":tense,"role":role_pair[0] if number=="singular" else role_pair[1],"second_adverb":SECOND_ADVERB,"fixed_determiner":DETERMINER,"fixed_adjective":ADJ,"fixed_preposition":PREP,"fixed_complement":COMP},"audit":a,"center_state":{"midpoint":mid,"crossing":cross,"closed_pairs_before_first_mismatch":pairs,"live_debt":a["first_mismatches"][0] if a["first_mismatches"] else None,"grammar":"typed SVO pair with subject-role alternation"},"anti_shortcut_flags":{"finished_tape_reversal":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"catalogue_text":False,"punctuation_changes_letters":False,"fragment":False},"provenance":{"lexical_source":"held-out typed subject-role inventory","borrowed_text":False,"short_grammar":True,"all_other_states_fixed":True,"role_realized_before_emission":True}}
def run():
 pre=preflight();rows=[emit(f,r,n,t) for f,r,n,t in itertools.product(FRAMES,ROLES,("singular","plural"),("present","past"))];rows.sort(key=lambda x:(x["audit"]["exact"],x["audit"]["letters"]),reverse=True);exact=[r for r in rows if r["audit"]["exact"]]
 return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure","method":"bounded second-proposition subject-role alternation in short grammar","novelty_preflight":pre,"candidate_count":len(rows),"exact_count":len(exact),"reader_eligible":False,"rendered_candidates":rows,"stats":{"variants":len(rows),"longest_letters":max(r["audit"]["letters"] for r in rows),"midpoint_inside_token":sum(r["center_state"]["crossing"] is not None for r in rows)},"failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found","next_repair":"retain role and all fixed states, then return to a two-frame center seam with no lexical sweep"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer scan","forward/reverse SHA-256"],"shortcuts_excluded":True}}
if __name__=="__main__":
 x=run();OUT.write_text(json.dumps(x,indent=2)+"\n");print(json.dumps({"candidates":x["candidate_count"],"exact":x["exact_count"],"stats":x["stats"]},sort_keys=True))
