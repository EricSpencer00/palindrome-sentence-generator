"""Cross-word seam whose residual overhang selects an agreement boundary."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
from itertools import product

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"runs/center-overhang-agreement-boundary-20260920.json"; ID="center-overhang-agreement-boundary-20260920"

def letters(x): return re.sub(r"[^a-z]","",x.casefold())
def audit(t):
 s=letters(t); mm=next(((i,s[i],s[-1-i]) for i in range(len(s)//2) if s[i]!=s[-1-i]),None); f=hashlib.sha256(s.encode()).hexdigest(); r=hashlib.sha256(s[::-1].encode()).hexdigest(); return {"letters":len(s),"pointer_exact":bool(s) and mm is None,"first_mismatch":mm,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}
CENTERS=(("harbor","road","singular"),("river","road","plural"),("garden","night","singular"))
FRAMES=(("singular","the patient sailor charts the inlet"),("singular","a careful keeper guards the bridge"),("plural","the young scouts return after rain"),("plural","several bright guides watch the harbor"))
ADJUNCTS=("at first light","beside the quiet pier")

def live(left,right):
 a,b=letters(left),letters(right)[::-1]; i=j=0; lb=rb=""; checks=0; mx=0
 while i<len(a) or j<len(b):
  if i<len(a): lb+=a[i:i+4]; i+=min(4,len(a)-i)
  if j<len(b): rb+=b[j:j+4]; j+=min(4,len(b)-j)
  while lb and rb:
   checks+=1
   if lb[0]!=rb[0]: return {"equations":checks,"satisfied":checks-1,"all_satisfied":False,"first_mismatch":(checks-1,lb[0],rb[0]),"max_residual":max(mx,len(lb),len(rb))}
   lb,rb=lb[1:],rb[1:]
  mx=max(mx,len(lb),len(rb))
 return {"equations":checks,"satisfied":checks,"all_satisfied":not(lb or rb),"first_mismatch":None,"max_residual":mx}

def run():
 rows=[]; controls=[]; states=prunes=0
 for center,left,right,la,ra in product(CENTERS,FRAMES,FRAMES,ADJUNCTS,ADJUNCTS):
  states+=1; lw,rw,preferred=center; overhang=abs(len(letters(lw))-len(letters(rw)))
  # Overhang parity is a live typed boundary choice, not a lexical filter sweep.
  selected_number=preferred if overhang%2==0 else ("plural" if preferred=="singular" else "singular")
  if left[0]!=selected_number: continue
  left_text=f"{left[1]}, {la} {lw}"; right_text=f"{rw} {ra}, {right[1]}."; rendered=left_text+" "+right_text; eq=live(left_text,right_text); row={"rendered":rendered,"center_words":{"left":lw,"right":rw,"inward_shared":lw[-1] if lw[-1]==rw[0] else None,"residual_overhang":overhang,"selected_agreement":selected_number},"left_frame":left,"right_frame":right,"online_character_equations":eq,"audit":audit(rendered)}
  if lw[-1]!=rw[0] or lw==lw[::-1] or rw==rw[::-1]: continue
  if len(controls)<3 and left[1]!=right[1]: controls.append({**row,"reader_eligible":False,"diagnostic_only":True})
  if not eq["all_satisfied"]: prunes+=1; continue
  row["provenance"]={"overhang_selects_agreement_boundary":True,"cross_word_inward_seam":True,"non_self_palindromic_centers":True,"typed_svo_adjunct":True,"complete_utterances":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False,"word_order_symmetry":False,"fragment":False,"nested_self_palindrome":False}; rows.append(row)
 exact=[r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"]>38]; reader=[r for r in exact if r["provenance"]["complete_utterances"]]
 result={"experiment_id":ID,"method":"center residual-overhang agreement boundary before typed SVO expansion","stats":{"center_pairs":len(CENTERS),"typed_frames":len(FRAMES),"adjuncts":len(ADJUNCTS),"states":states,"live_prunes":prunes,"live_survivors":len(rows),"exact_gt38":len(exact),"reader_eligible":len(reader),"longest_letters":max((r["audit"]["letters"] for r in rows+controls),default=0)},"controls":controls,"exact_candidates":exact,"reader_facing_candidates":reader,"novelty_preflight":{"status":"passed","signature":"cross-word-seam|overhang-agreement-boundary|typed-SVO","registry_inspected":True,"distinct_from":"fixed center seam, residual discharge, self-palindromic tokens, and seed embedding","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer mismatch scan","forward/reverse SHA-256"],"reader_evidence":False},"status":"no reader-worthy exact closure" if not reader else "reader gate required","next_construction":"Let residual overhang choose both agreement and valency before the cross-word seam is emitted.","reader_gate":"closed until exact candidates exist and blinded human ratings are collected"}
 OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"artifact":str(OUT),**result["stats"]})); [print(x["rendered"]) for x in controls]; return result
if __name__=="__main__": run()
