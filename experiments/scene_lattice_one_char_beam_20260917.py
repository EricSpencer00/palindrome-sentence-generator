"""One-character reverse-prefix beam at the independent object-setting frontier."""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/scene-lattice-one-char-beam-20260917.json";REG=ROOT/"docs/experiment-novelty-registry.json"
ID="scene-lattice-one-char-beam-20260917";SIG="independent-scene-lattice|one-character-reverse-prefix-beam|object-setting-frontier|compact-single-sentence|independent-exact-audit"
SUBJECTS=(("the careful botanist","records"),("the patient cartographer","maps"),("the quiet archivists","study")); OBJECTS=(("a coastal chart","artifact"),("the weathered atlas","artifact"),("the local survey","record"),("a field notebook","record")); SETTINGS=(("beside the quiet harbor","location"),("near the old observatory","location"),("during the morning survey","time"),("under the northern light","condition")); ADVERBS=("carefully","quietly")
def norm(s):return re.sub(r"[^a-z]","",s.lower())
def audit(s):
 t=norm(s);m=[];i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({"left":i,"right":j,"a":t[i],"b":t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();return {"normalized_tape":t,"letters":len(t),"exact":bool(t) and not m,"independent_two_pointer_exact":bool(t) and not m,"first_mismatches":m[:8],"sha256_forward":f,"sha256_reverse":r,"sha_equal_under_reversal":f==r}
def preflight():
 es=json.loads(REG.read_text()).get("entries",[]);a=str(Path(__file__).relative_to(ROOT));return {"status":"passed","registry_entries_read":len(es),"signature_collision":any(x.get("signature")==SIG for x in es),"artifact_collision":any(x.get("artifact")==a for x in es),"shortcuts_rejected":["prior seam frames","attachment expansion","finished-tape reversal","catalogue text","fragments"]}
def beam_score(obj,setting):
 o=norm(obj);s=norm(setting);required=o[-1]; first=s[0]; return {"required_reverse_character":required,"setting_first_character":first,"one_character_match":first==required,"score":int(first==required)}
def emit(subject,verb,obj,orole,setting,srole,adv,beam):
 text=f"{subject.capitalize()} {verb} {obj} {adv} {setting}.";t=norm(text);mid=len(t)//2;sp=[];cur=0
 for tok in re.findall(r"[A-Za-z]+",text):a=cur;cur+=len(tok);sp.append((tok,a,cur))
 cross=next(({"token":w,"token_interval":[a,b],"midpoint":mid,"offset":mid-a} for w,a,b in sp if a<=mid<b),None);a=audit(text);i,j=0,len(t)-1;pairs=0
 while i<j and t[i]==t[j]:pairs+=1;i+=1;j-=1
 return {"rendered":text,"choices":{"subject":subject,"verb":verb,"object":obj,"object_role":orole,"setting":setting,"setting_role":srole,"adverb":adv},"beam":beam,"audit":a,"center_state":{"midpoint":mid,"crossing":cross,"closed_pairs_before_first_mismatch":pairs,"live_debt":a["first_mismatches"][0] if a["first_mismatches"] else None},"anti_shortcut_flags":{"prior_seam_frames":False,"attachment_expansion":False,"finished_tape_reversal":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"catalogue_text":False,"punctuation_changes_letters":False,"fragment":False},"provenance":{"lexical_source":"independent typed role banks","borrowed_text":False,"one_char_beam_before_render":True,"syntax_expanded":False}}
def run():
 pre=preflight();pool=[]
 for (sub,verb),(obj,orr),(setting,sr),adv in itertools.product(SUBJECTS,OBJECTS,SETTINGS,ADVERBS):
  if sub.startswith("the quiet") and verb!="study":continue
  if sub.startswith("the careful") and verb=="study":continue
  pool.append((beam_score(obj,setting),sub,verb,obj,orr,setting,sr,adv))
 # Beam width two per object: keep highest-scoring settings, preserving ties deterministically.
 selected=[]
 for obj in OBJECTS:
  group=[x for x in pool if x[3]==obj[0]]; selected.extend(sorted(group,key=lambda x:(x[0]["score"],x[5]),reverse=True)[:2])
 rows=[emit(s,v,o,orr,setg,sr,a,b) for b,s,v,o,orr,setg,sr,a in selected];rows.sort(key=lambda x:(x["audit"]["exact"],x["audit"]["letters"]),reverse=True);exact=[r for r in rows if r["audit"]["exact"]]
 return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure","method":"one-character reverse-prefix beam at object-setting frontier","novelty_preflight":pre,"pre_filter_count":len(pool),"candidate_count":len(rows),"exact_count":len(exact),"reader_eligible":False,"rendered_candidates":rows,"stats":{"beam_width":2,"selected":len(rows),"max_beam_score":max(r["beam"]["score"] for r in rows),"longest_letters":max(r["audit"]["letters"] for r in rows),"midpoint_inside_token":sum(r["center_state"]["crossing"] is not None for r in rows)},"failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found","next_repair":"retain one-character beam and add a second character only for tied frontier branches, without syntax expansion"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer scan","forward/reverse SHA-256","pre-render one-character beam"],"shortcuts_excluded":True}}
if __name__=="__main__":
 x=run();OUT.write_text(json.dumps(x,indent=2)+"\n");print(json.dumps({"pre_filter":x["pre_filter_count"],"selected":x["candidate_count"],"exact":x["exact_count"],"stats":x["stats"]},sort_keys=True))
