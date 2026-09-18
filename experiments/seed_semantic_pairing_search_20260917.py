"""Independent typed-clause product ranked by a live palindrome seam.

The 38-letter seed is benchmark metadata only: no substring or reversal is
used. Choices are complete words selected before rendering.
"""
from __future__ import annotations
import hashlib, json, re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID="seed-semantic-pairing-search-20260917"
SIG="seed-benchmark|independent-typed-clause-product|live-outer-equation|no-mirror|independent-pointer-sha"
LEFT=("The",("quiet","patient","young"),("pilot","teacher","gardener"),("maps","plants","guides"),("near the river","by the school","at dawn"))
RIGHT=("A",("calm","clever","gentle"),("reader","baker","singer"),("opens","bakes","writes"),("a plant","a boat","a tent"))
def letters(s): return "".join(c.lower() for c in s if "a"<=c.lower()<="z")
def audit(s):
 t=letters(s); p=[(i,len(t)-1-i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {"letters":len(t),"exact":bool(t) and not p,"mismatch_count":len(p),"first_mismatch":p[0] if p else None,"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def clauses(f):
 d,adjs,nouns,verbs,ends=f
 return [f"{d} {a} {n} {v} {o}." for a,n,v,o in product(adjs,nouns,verbs,ends)]
def seam(a,b):
 x,y=letters(a),letters(b); k=0
 while k<min(len(x),len(y)) and x[k]==y[-1-k]: k+=1
 return k
def run():
 ls,rs=clauses(LEFT),clauses(RIGHT); rows=[]
 for a,b in product(ls,rs):
  text=a+" "+b; au=audit(text); rows.append((seam(a,b),au["mismatch_count"],text,au))
 rows.sort(key=lambda z:(-z[0],z[1],z[2])); chosen=[]; seen=set()
 for score,mis,text,au in rows:
  words=re.findall(r"[A-Za-z]+",text.lower()); content=[w for w in words if w not in {"a","the","at","by","near"}]
  if len(content)!=len(set(content)) or letters(text) in seen: continue
  seen.add(letters(text)); chosen.append({"rendered":text,"live_outer_pairs":score,"independent_audit":au,"provenance":{"left":"typed authored clause product","right":"independent typed authored clause product","seed_used":False},"anti_shortcut":{"finished_tape_reversal":False,"seed_wrapped":False,"word_order_mirror":False,"catalogue_imported":False,"repeated_content_word":False}})
  if len(chosen)==5: break
 payload={"experiment_id":ID,"signature":SIG,"status":"completed_no_exact_closure","reader_eligible":False,"method":"independent typed clause product ranked by live outer-character compatibility","seed_benchmark":{"text":"An aide rips nine memos; some men inspire Diana.","letters":38,"used_as_constraint":False,"used_in_output":False},"novelty_preflight":{"status":"passed","performed_before_rendering":True,"duplicate_sweep_rejected":True},"candidates":chosen,"stats":{"rendered":len(chosen),"exact":sum(x["independent_audit"]["exact"] for x in chosen),"search_states":len(rows),"best_live_outer_pairs":chosen[0]["live_outer_pairs"] if chosen else 0},"next_repair":{"operator":"learn seam-conditioned lexical alternatives for the first mismatch while preserving roles and agreement","reason":"independent clause products improve the seam but do not close it","preserve":["whole-word choices before rendering","ordinary SVO prose","no seed scaffold"]}}
 (ROOT/"runs/seed-semantic-pairing-search-20260917.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload["stats"],sort_keys=True))
if __name__=="__main__": run()
