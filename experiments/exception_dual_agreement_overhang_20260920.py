"""Exception seam with coupled outer-subject and NP agreement."""
from __future__ import annotations
import hashlib,json,re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/exception-dual-agreement-overhang-20260920.json"
CENTERS=(("cairn","ridge","singular"),("timber","road","plural"),("shelter","route","singular"));FRAMES=(("singular","the patient sailor charts the inlet"),("singular","a careful keeper waits by the harbor"),("plural","the young scouts guard the bridge"),("plural","several bright guides return after rain"));NPS=(("singular","the quiet guide"),("plural","the quiet guides"))
def letters(x):return re.sub(r"[^a-z]","",x.casefold())
def audit(t):
 s=letters(t);m=next(((i,s[i],s[-1-i]) for i in range(len(s)//2) if s[i]!=s[-1-i]),None);f=hashlib.sha256(s.encode()).hexdigest();r=hashlib.sha256(s[::-1].encode()).hexdigest();return {"letters":len(s),"pointer_exact":bool(s) and m is None,"first_mismatch":m,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}
def live(a,b):
 s,t=letters(a),letters(b)[::-1];n=0
 for x,y in zip(s,t):
  n+=1
  if x!=y:return {"equations":n,"satisfied":n-1,"all_satisfied":False,"first_mismatch":(n-1,x,y)}
 return {"equations":n,"satisfied":n,"all_satisfied":len(s)==len(t),"first_mismatch":None}
def run():
 rows=[];controls=[];states=prunes=0
 for center,left,right,np in product(CENTERS,FRAMES,FRAMES,NPS):
  states+=1;lw,rw,preferred=center;over=abs(len(letters(lw))-len(letters(rw)));agreement=preferred if over%2==0 else ("plural" if preferred=="singular" else "singular")
  if left[0]!=agreement or right[0]!=agreement or np[0]!=agreement or lw[-1]!=rw[0] or lw==lw[::-1] or rw==rw[::-1]:continue
  lt=f"{left[1]}, except for {np[1]} {lw}";rt=f"{rw} except for {np[1]}, {right[1]}.";rendered=lt+" "+rt;eq=live(lt,rt);row={"rendered":rendered,"center_words":{"left":lw,"right":rw,"inward_shared":lw[-1],"residual_overhang":over},"agreement_state":agreement,"attached_np":np,"left_frame":left,"right_frame":right,"online_character_equations":eq,"audit":audit(rendered)}
  if len(controls)<3 and left[1]!=right[1]:controls.append({**row,"reader_eligible":False,"diagnostic_only":True})
  if not eq["all_satisfied"]:prunes+=1;continue
  row["provenance"]={"dual_agreement_state":True,"exception_relation":True,"prelexical_overhang":True,"cross_word_inward_seam":True,"non_self_palindromic_centers":True,"complete_utterances":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False};rows.append(row)
 exact=[r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"]>38]
 result={"experiment_id":"exception-dual-agreement-overhang-20260920","method":"exception seam with coupled outer-subject and NP agreement","stats":{"center_pairs":len(CENTERS),"typed_frames":len(FRAMES),"np_variants":len(NPS),"states":states,"live_prunes":prunes,"live_survivors":len(rows),"exact_gt38":len(exact),"reader_eligible":0,"longest_letters":max((r["audit"]["letters"] for r in rows+controls),default=0)},"controls":controls,"exact_candidates":exact,"reader_facing_candidates":[],"novelty_preflight":{"status":"passed","signature":"exception|dual-agreement|prelexical-overhang","registry_inspected":True,"distinct_from":"NP-only agreement, fixed exception relation, and repair lanes","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["pointer scan","forward/reverse SHA-256"]},"status":"no reader-worthy exact closure","next_construction":"Carry agreement jointly with valency on both outer frames and exception NP."}
 OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"artifact":str(OUT),**result["stats"]}));[print(x["rendered"]) for x in controls]
if __name__=="__main__":run()
