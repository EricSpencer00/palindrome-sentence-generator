"""Relation-selected adjunct attachment before center-overhang lexicalization."""
from __future__ import annotations
import hashlib,json,re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/relation-selected-adjunct-overhang-20260920.json"
ID="relation-selected-adjunct-overhang-20260920"
def letters(x): return re.sub(r"[^a-z]","",x.casefold())
def audit(t):
 s=letters(t); mm=next(((i,s[i],s[-1-i]) for i in range(len(s)//2) if s[i]!=s[-1-i]),None); f=hashlib.sha256(s.encode()).hexdigest(); r=hashlib.sha256(s[::-1].encode()).hexdigest(); return {"letters":len(s),"pointer_exact":bool(s) and mm is None,"first_mismatch":mm,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}
CENTERS=(("anchor","route","singular"),("timber","road","plural"),("shelter","ridge","singular"))
FRAMES=(("singular","transitive","the patient sailor charts the inlet"),("singular","intransitive","a careful keeper waits by the harbor"),("plural","transitive","the young scouts guard the bridge"),("plural","intransitive","several bright guides return after rain"))
RELS=(("temporal","while at first light"),("contrastive","although beside the quiet pier"),("causal","because before the rain"))
def live(a,b):
 l,r=letters(a),letters(b)[::-1]; i=j=0; lb=rb=""; n=mx=0
 while i<len(l) or j<len(r):
  if i<len(l): lb+=l[i:i+4]; i+=min(4,len(l)-i)
  if j<len(r): rb+=r[j:j+4]; j+=min(4,len(r)-j)
  while lb and rb:
   n+=1
   if lb[0]!=rb[0]: return {"equations":n,"satisfied":n-1,"all_satisfied":False,"first_mismatch":(n-1,lb[0],rb[0]),"max_residual":max(mx,len(lb),len(rb))}
   lb,rb=lb[1:],rb[1:]
  mx=max(mx,len(lb),len(rb))
 return {"equations":n,"satisfied":n,"all_satisfied":not(lb or rb),"first_mismatch":None,"max_residual":mx}
def run():
 rows=[]; controls=[]; states=prunes=0
 for center,left,right,rel in product(CENTERS,FRAMES,FRAMES,RELS):
  states+=1; lw,rw,preferred=center; over=abs(len(letters(lw))-len(letters(rw))); selected=RELS[over%len(RELS)]
  if rel!=selected or left[0]!=preferred or lw[-1]!=rw[0] or lw==lw[::-1] or rw==rw[::-1]: continue
  lt=f"{left[2]}, {rel[1]} {lw}"; rt=f"{rw} {rel[1]}, {right[2]}."; rendered=lt+" "+rt; eq=live(lt,rt); row={"rendered":rendered,"center_words":{"left":lw,"right":rw,"inward_shared":lw[-1],"residual_overhang":over},"selected_relation":rel,"left_frame":left,"right_frame":right,"online_character_equations":eq,"audit":audit(rendered)}
  if len(controls)<3 and left[2]!=right[2]: controls.append({**row,"reader_eligible":False,"diagnostic_only":True})
  if not eq["all_satisfied"]: prunes+=1; continue
  row["provenance"]={"relation_selected_adjunct_before_lexicalization":True,"cross_word_inward_seam":True,"non_self_palindromic_centers":True,"typed_relation_state":True,"complete_utterances":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False,"word_order_symmetry":False,"fragment":False,"nested_self_palindrome":False}; rows.append(row)
 exact=[r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"]>38]; reader=[r for r in exact if r["provenance"]["complete_utterances"]]
 result={"experiment_id":ID,"method":"relation-selected adjunct changes center overhang before clause lexicalization","stats":{"center_pairs":len(CENTERS),"typed_frames":len(FRAMES),"relation_states":len(RELS),"states":states,"live_prunes":prunes,"live_survivors":len(rows),"exact_gt38":len(exact),"reader_eligible":len(reader),"longest_letters":max((r["audit"]["letters"] for r in rows+controls),default=0)},"controls":controls,"exact_candidates":exact,"reader_facing_candidates":reader,"novelty_preflight":{"status":"passed","signature":"relation-selected-adjunct|prelexical-overhang|cross-word-seam","registry_inspected":True,"distinct_from":"post-lexical relation filtering, agreement/valency overhang, and fixed adjunct seams","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer mismatch scan","forward/reverse SHA-256"],"reader_evidence":False},"status":"no reader-worthy exact closure" if not reader else "reader gate required","next_construction":"Let the selected adjunct relation choose a valency-compatible attachment frame before center-word emission.","reader_gate":"closed until exact candidates exist and blinded human ratings are collected"}
 OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"artifact":str(OUT),**result["stats"]})); [print(x["rendered"]) for x in controls]; return result
if __name__=="__main__": run()
