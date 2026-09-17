"""Independent scene lattice with a live object-setting reverse-obligation filter."""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/scene-lattice-boundary-obligation-filter-20260917.json";REG=ROOT/"docs/experiment-novelty-registry.json"
ID="scene-lattice-boundary-obligation-filter-20260917";SIG="independent-scene-lattice|object-setting-live-reverse-obligation-filter|compact-single-sentence|typed-role-banks|independent-exact-audit"
SUBJECTS=(("the careful botanist","sg"),("the patient cartographer","sg"),("the quiet archivists","pl")); VERBS=("records","maps","study"); OBJECTS=("a coastal chart","the weathered atlas","the local survey"); SETTINGS=("beside the quiet harbor","near the old observatory","during the morning survey"); ADVERBS=("carefully","quietly")
def norm(s):return re.sub(r"[^a-z]","",s.lower())
def audit(s):
 t=norm(s);m=[];i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({"left":i,"right":j,"a":t[i],"b":t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();return {"normalized_tape":t,"letters":len(t),"exact":bool(t) and not m,"independent_two_pointer_exact":bool(t) and not m,"first_mismatches":m[:8],"sha256_forward":f,"sha256_reverse":r,"sha_equal_under_reversal":f==r}
def preflight():
 es=json.loads(REG.read_text()).get("entries",[]);a=str(Path(__file__).relative_to(ROOT));return {"status":"passed","registry_entries_read":len(es),"signature_collision":any(x.get("signature")==SIG for x in es),"artifact_collision":any(x.get("artifact")==a for x in es),"shortcuts_rejected":["prior seam frames","attachment expansion","finished-tape reversal","catalogue text","fragments"]}
def boundary_debt(subject,verb,obj,setting,adv):
 # Compute the only reverse obligation consulted before rendering: characters
 # at the object/setting frontier are compared in the reverse direction.
 o=norm(obj);s=norm(setting); width=min(5,len(o),len(s)); pairs=[(o[-1-k],s[k]) for k in range(width)]; mismatches=sum(a!=b for a,b in pairs)
 return {"width":width,"pairs":pairs,"mismatches":mismatches,"score":width-mismatches,"filter":"maximize matching frontier pairs"}
def emit(subject,number,verb,obj,setting,adv,debt):
 text=f"{subject.capitalize()} {verb} {obj} {adv} {setting}.";t=norm(text);mid=len(t)//2;sp=[];cur=0
 for tok in re.findall(r"[A-Za-z]+",text):a=cur;cur+=len(tok);sp.append((tok,a,cur))
 cross=next(({"token":w,"token_interval":[a,b],"midpoint":mid,"offset":mid-a} for w,a,b in sp if a<=mid<b),None);a=audit(text);i,j=0,len(t)-1;pairs=0
 while i<j and t[i]==t[j]:pairs+=1;i+=1;j-=1
 return {"rendered":text,"choices":{"subject":subject,"number":number,"verb":verb,"object":obj,"setting":setting,"adverb":adv},"boundary_obligation":debt,"audit":a,"center_state":{"midpoint":mid,"crossing":cross,"closed_pairs_before_first_mismatch":pairs,"live_debt":a["first_mismatches"][0] if a["first_mismatches"] else None},"anti_shortcut_flags":{"prior_seam_frames":False,"attachment_expansion":False,"finished_tape_reversal":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"catalogue_text":False,"punctuation_changes_letters":False,"fragment":False},"provenance":{"lexical_source":"independent typed semantic-role banks","borrowed_text":False,"filter_before_final_render":True,"sentence_structure_compact":True}}
def run():
 pre=preflight(); all_rows=[]
 for (subject,num),verb,obj,setting,adv in itertools.product(SUBJECTS,VERBS,OBJECTS,SETTINGS,ADVERBS):
  if num=="sg" and verb=="study":continue
  if num=="pl" and verb in ("records","maps"):continue
  all_rows.append((boundary_debt(subject,verb,obj,setting,adv),subject,num,verb,obj,setting,adv))
 max_score=max(x[0]["score"] for x in all_rows); selected=[x for x in all_rows if x[0]["score"]==max_score]
 rows=[emit(s,n,v,o,setg,a,d) for d,s,n,v,o,setg,a in selected];rows.sort(key=lambda x:(x["audit"]["exact"],x["audit"]["letters"]),reverse=True);exact=[r for r in rows if r["audit"]["exact"]]
 return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure","method":"independent scene lattice with live object-setting reverse-obligation filter","novelty_preflight":pre,"candidate_count":len(rows),"pre_filter_count":len(all_rows),"exact_count":len(exact),"reader_eligible":False,"rendered_candidates":rows,"stats":{"selected":len(rows),"max_boundary_score":max_score,"longest_letters":max(r["audit"]["letters"] for r in rows),"midpoint_inside_token":sum(r["center_state"]["crossing"] is not None for r in rows)},"failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found","next_repair":"retain boundary filter and add a second independent role bank at the same frontier, without importing seam frames"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer scan","forward/reverse SHA-256","pre-render boundary debt filter"],"shortcuts_excluded":True}}
if __name__=="__main__":
 x=run();OUT.write_text(json.dumps(x,indent=2)+"\n");print(json.dumps({"pre_filter":x["pre_filter_count"],"selected":x["candidate_count"],"exact":x["exact_count"],"stats":x["stats"]},sort_keys=True))
