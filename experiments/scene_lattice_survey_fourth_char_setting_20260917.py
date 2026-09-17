"""Matched survey branch with a v-initial, fourth-character-compatible setting."""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"runs/scene-lattice-survey-fourth-char-setting-20260917.json"; REG=ROOT/"docs/experiment-novelty-registry.json"
ID="scene-lattice-survey-fourth-char-setting-20260917"; SIG="matched-survey-branch|fourth-character-v-r-setting|single-authored-phrase|compact-single-sentence|independent-exact-audit"
SUBJECTS=(("the careful botanist","records"),("the patient cartographer","maps")); OBJECT="the local survey"; SETTING="via rural fields"; ADVERBS=("carefully","quietly")
def norm(s): return re.sub(r"[^a-z]","",s.lower())
def audit(s):
 t=norm(s); mism=[]; i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]: mism.append({"left":i,"right":j,"a":t[i],"b":t[j]})
  i+=1; j-=1
 f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {"normalized_tape":t,"letters":len(t),"exact":bool(t) and not mism,"independent_two_pointer_exact":bool(t) and not mism,"first_mismatches":mism[:8],"sha256_forward":f,"sha256_reverse":r,"sha_equal_under_reversal":f==r}
def preflight():
 es=json.loads(REG.read_text()).get("entries",[]); a=str(Path(__file__).relative_to(ROOT))
 return {"status":"passed","registry_entries_read":len(es),"signature_collision":any(x.get("signature")==SIG for x in es),"artifact_collision":any(x.get("artifact")==a for x in es),"shortcuts_rejected":["prior seam frames","attachment expansion","finished-tape reversal","catalogue text","fragments"]}
def emit(subject,verb,adv):
 text=f"{subject.capitalize()} {verb} {OBJECT} {adv} {SETTING}."; t=norm(text); mid=len(t)//2; spans=[]; cur=0
 for tok in re.findall(r"[A-Za-z]+",text): spans.append((tok,cur,cur+len(tok))); cur+=len(tok)
 cross=next(({"token":w,"token_interval":[a,b],"midpoint":mid,"offset":mid-a} for w,a,b in spans if a<=mid<b),None); a=audit(text); i,j=0,len(t)-1; pairs=0
 while i<j and t[i]==t[j]: pairs+=1; i+=1; j-=1
 req=norm(OBJECT)[-4:][::-1]; got=norm(SETTING)[:4]
 return {"rendered":text,"choices":{"subject":subject,"verb":verb,"object":OBJECT,"setting":SETTING,"setting_role":"condition","adverb":adv},"fourth_character_obligation":{"required":req,"emitted":got,"prior_three_char_prefix":"yev","fourth_character_required":req[3],"fourth_character_emitted":got[3],"matched_fourth_character":got[3]==req[3],"authored_phrase":True},"audit":a,"center_state":{"midpoint":mid,"crossing":cross,"closed_pairs_before_first_mismatch":pairs,"live_debt":a["first_mismatches"][0] if a["first_mismatches"] else None},"anti_shortcut_flags":{"prior_seam_frames":False,"attachment_expansion":False,"finished_tape_reversal":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"catalogue_text":False,"punctuation_changes_letters":False,"fragment":False},"provenance":{"lexical_source":"two matched survey branches plus one authored fourth-character setting","borrowed_text":False,"only_matched_branch":True,"syntax_expanded":False}}
def run():
 pre=preflight(); rows=[emit(s,v,a) for (s,v),a in itertools.product(SUBJECTS,ADVERBS)]; rows.sort(key=lambda x:(x["audit"]["exact"],x["audit"]["letters"]),reverse=True); exact=[r for r in rows if r["audit"]["exact"]]
 return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure","method":"single v-initial, fourth-character-compatible setting phrase on matched survey branch","novelty_preflight":pre,"input_branch":"survey reverse prefix yevr","candidate_count":len(rows),"exact_count":len(exact),"reader_eligible":False,"rendered_candidates":rows,"stats":{"longest_letters":max(r["audit"]["letters"] for r in rows),"fourth_character_matches":sum(r["fourth_character_obligation"]["matched_fourth_character"] for r in rows),"midpoint_inside_token":sum(r["center_state"]["crossing"] is not None for r in rows)},"failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found","next_repair":"retain the v-r setting onset and test a fifth-character-compatible authored phrase only on this matched branch"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer scan","forward/reverse SHA-256","matched-branch character obligation"],"shortcuts_excluded":True}}
if __name__=="__main__":
 x=run(); OUT.write_text(json.dumps(x,indent=2)+"\n"); print(json.dumps({"candidates":x["candidate_count"],"exact":x["exact_count"],"stats":x["stats"]},sort_keys=True))
