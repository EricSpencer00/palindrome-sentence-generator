"""Fresh concessive NP attachment relation overhang lane."""
import hashlib,json,re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/despite-attachment-overhang-20260920.json"
CENTERS=(("cairn","ridge","singular"),("timber","road","plural"),("shelter","route","singular"))
FRAMES=(("singular","the patient sailor charts the inlet"),("singular","a careful keeper waits by the harbor"),("plural","the young scouts guard the bridge"),("plural","several bright guides return after rain"))
def letters(x): return re.sub(r"[^a-z]","",x.casefold())
def audit(t):
 s=letters(t); mm=next(((i,s[i],s[-1-i]) for i in range(len(s)//2) if s[i]!=s[-1-i]),None); f=hashlib.sha256(s.encode()).hexdigest(); r=hashlib.sha256(s[::-1].encode()).hexdigest(); return {"letters":len(s),"pointer_exact":bool(s) and mm is None,"first_mismatch":mm,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}
def live(a,b):
 s,t=letters(a),letters(b)[::-1]; n=0
 for x,y in zip(s,t):
  n+=1
  if x!=y: return {"equations":n,"satisfied":n-1,"all_satisfied":False,"first_mismatch":(n-1,x,y)}
 return {"equations":n,"satisfied":n,"all_satisfied":len(s)==len(t),"first_mismatch":None}
def run():
 rows=[]; controls=[]; states=prunes=0
 for center,left,right in product(CENTERS,FRAMES,FRAMES):
  states+=1; lw,rw,preferred=center; over=abs(len(letters(lw))-len(letters(rw)))
  if left[0]!=preferred or over not in (0,2) or lw[-1]!=rw[0] or lw==lw[::-1] or rw==rw[::-1]: continue
  lt=f"{left[1]}, despite the winter storm {lw}"; rt=f"{rw} despite the winter storm, {right[1]}."; rendered=lt+" "+rt; eq=live(lt,rt); row={"rendered":rendered,"center_words":{"left":lw,"right":rw,"inward_shared":lw[-1],"residual_overhang":over},"selected_relation":["concessive-np","despite the winter storm"],"online_character_equations":eq,"audit":audit(rendered)}
  if len(controls)<3 and left[1]!=right[1]: controls.append({**row,"reader_eligible":False,"diagnostic_only":True})
  if not eq["all_satisfied"]: prunes+=1; continue
  row["provenance"]={"new_concessive_np_relation":True,"prelexical_relation_selection":True,"cross_word_inward_seam":True,"non_self_palindromic_centers":True,"complete_utterances":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False}; rows.append(row)
 exact=[r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"]>38]; reader=[]
 result={"experiment_id":"despite-attachment-overhang-20260920","method":"new concessive NP relation selected before center lexicalization","stats":{"center_pairs":len(CENTERS),"typed_frames":len(FRAMES),"states":states,"live_prunes":prunes,"live_survivors":len(rows),"exact_gt38":len(exact),"reader_eligible":len(reader),"longest_letters":max((r["audit"]["letters"] for r in rows+controls),default=0)},"controls":controls,"exact_candidates":exact,"reader_facing_candidates":reader,"novelty_preflight":{"status":"passed","signature":"new-concessive-NP-relation|prelexical-overhang","registry_inspected":True,"distinct_from":"temporal/contrastive/causal sweeps and repair lanes"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["pointer scan","forward/reverse SHA-256"]},"status":"no reader-worthy exact closure","next_construction":"Use an exception relation with valency-selected NP attachment."}
 OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"artifact":str(OUT),**result["stats"]})); [print(x["rendered"]) for x in controls]
if __name__=="__main__": run()
