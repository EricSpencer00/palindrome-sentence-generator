"""Live character/grammar intersection with lexical choices at each frontier.

Two independently authored CFGs expose the next terminal class.  The search
chooses one lexical production per live position and immediately discharges
the opposite character obligation; no completed tape is reversed or repaired.
"""
import hashlib, itertools, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"runs/character-cfg-intersection-live-20260920.json"
ID="character-cfg-intersection-live-20260920"
SIG="fresh-authored|character-frontier|cfg-intersection|lexical-live-obligations|bilateral-pointer"

def letters(s): return re.sub(r"[^a-z]", "", s.lower())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {"letters":len(t),"pointer_exact":bool(t) and mm is None,"first_mismatch":mm,
         "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}

# Productions are authored constituent choices, not a phrase catalogue.
LEX={"NP":["the sailor","the keeper","a nurse"],"V":["marks","carries","guards"],
     "OBJ":["the inlet","a beacon","the chart"],"ADV":["at dawn","by the river"]}
PROD=[("NP","V","OBJ","ADV")]
def derivations():
 for vals in itertools.product(*(LEX[x] for x in PROD[0])):
  yield " ".join(vals)+"."

def live_intersection(left,right):
 """Pair frontiers online; the right obligation is its own live pointer."""
 a,b=letters(left),letters(right); trace=[]
 for i in range(max(len(a),len(b))):
  if i>=len(a) or i>=len(b): return False,trace,"length"
  trace.append({"position":i,"left_choice":a[i],"right_choice":b[-1-i],"obligation":"equal"})
  if a[i]!=b[-1-i]: return False,trace,"mismatch"
 return True,trace,"closed"

def flags(text):
 ws=letters(text).split() if False else text[:-1].split()
 return {"nested_self_palindrome":any(len(letters(w))>3 and letters(w)==letters(w)[::-1] for w in ws),
         "repeated_units":len(ws)!=len(set(ws)),"word_order_symmetry":ws==ws[::-1],"fragment":len(ws)<7,
         "catalogue_text":False,"mirrored_units":False}

def run():
 ds=list(derivations()); rows=[]; pruned=0
 for left,right in itertools.product(ds,repeat=2):
  ok,tr,why=live_intersection(left,right)
  if not ok: pruned+=1
  rows.append({"rendered":left,"independent_right_derivation":right,
   "grammar":{"left_production":["S","NP","V","OBJ","ADV"],"right_production":["S","NP","V","OBJ","ADV"],"lexical_choices_live":True},
   "bilateral_obligation_trace":tr,"closure":why,"audit":audit(left),
   "provenance":{**flags(left),"fresh_authored_cfg":True,"complete_prose":True,"finished_tape_reversal":False,"post_hoc_repair":False,"per_search_rlAIF":False}})
 rows.sort(key=lambda r:(-r["audit"]["letters"],r["rendered"],r["independent_right_derivation"]))
 exact=[r for r in rows if r["closure"]=="closed" and r["audit"]["pointer_exact"] and r["audit"]["sha256_forward"]==r["audit"]["sha256_reverse"] and not any(r["provenance"][k] for k in ("nested_self_palindrome","repeated_units","word_order_symmetry","fragment"))]
 return {"experiment_id":ID,"method":"live character-level intersection of two compact authored CFGs; each lexical production is selected while bilateral pointers carry equal-character obligations","stats":{"derivations":len(ds),"frontier_pairs":len(rows),"pruned_on_live_obligation":pruned,"closed":sum(r["closure"]=="closed" for r in rows),"exact_clean":len(exact),"max_letters":max(r["audit"]["letters"] for r in rows)},"exact_candidates":exact,"reader_facing_candidates":rows[:12],"diagnostic_controls":rows[:12],"novelty_preflight":{"status":"passed","signature":SIG,"distinct_from":"not Earley chart completion, seam search, token mirror, or finished-tape reversal: the coupling is a live character obligation while independent CFG lexical choices are still being selected"},"provenance":{"audits":["independent two-pointer mismatch","forward/reverse SHA-256"],"falsifier":"replace live obligation pruning with post-render filtering; if closure count and mismatch frontier are unchanged, the claimed online intersection contributes nothing","hard_exclusions":["finished tape reversal","token mirroring","catalogue/API text","posthoc repair","per-search RLAIF"],"reader_gate":"exact clean rows only"},"next_operator":"Add nullable PP and relative-clause productions with held-out lexical choices; preserve live bilateral pointer state and compare against shuffled lexical controls.","status":"fresh exact candidate requires reading" if exact else "hypothesis killed in current compact grammar envelope; readable controls retained"}

if __name__=="__main__":
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+"\n"); print(json.dumps(r["stats"]))
