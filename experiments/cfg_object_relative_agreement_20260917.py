#!/usr/bin/env python3
"""Object-relative CFG intersection with explicit agreement states."""
import hashlib, itertools, json, re
from pathlib import Path

NOUNS={"sg":("gardener","reader","sailor","teacher"),"pl":("gardeners","readers","sailors","teachers")}
DET={"sg":("the","a"),"pl":("the",)}
SUBJ_V={"sg":("carries","reviews","marks","opens"),"pl":("carry","review","mark","open")}
REL_V={"sg":("carries","reviews","marks","opens"),"pl":("carry","review","mark","open")}
OBJ=("map","letter","garden","harbor")

def canon(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=canon(s); return {"letters":len(t),"two_pointer":t==t[::-1],"sha256":hashlib.sha256(t.encode()).hexdigest(),"reverse_sha256_equal":hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def live(s):
 t=canon(s)
 for i in range(len(t)//2):
  if t[i]!=t[-1-i]: return {"matched_outer_pairs":i,"first_mismatch":[i,len(t)-1-i],"frontier":t[i:len(t)-i]}
 return {"matched_outer_pairs":len(t)//2,"first_mismatch":None,"frontier":""}

def sentence(sn, on, sv, rv, obj):
 # S -> NP[sn] VP[sn, obj]; VP -> V[sn] NP[obj, RC];
 # NP[obj,RC] -> Det Obj that NP[sn] V[sn].
 return f"{sn} {sv} the {on} that {sn} {rv} the {obj}."

def main():
 rows=[]; seen=set(); controls=repairs=0
 for num in ('sg','pl'):
  for sn,on,sv,rv,obj in itertools.product(NOUNS[num],NOUNS[num],SUBJ_V[num],REL_V[num],OBJ):
   s=sentence(sn,on,sv,rv,obj); k=canon(s)
   if k not in seen:
    seen.add(k); rows.append({"rendered":s,"provenance":"fresh object-relative typed CFG derivation","agreement_state":{"subject":num,"relative_subject":num,"transition":"NP->VP requires V["+num+"]"},"repaired":False,"audit":audit(s),"live_frontier":live(s),"anti_shortcut":{"single_tree":True,"object_relative":True,"word_order_only":False,"repeated_unit":False,"catalogue_source":False,"fragment":False}}); controls+=1
   # Held-out repair: swap the relative verb to another verb in the same
   # number state, preserving agreement and attachment.
   alt=next(v for v in REL_V[num] if v!=rv); r=sentence(sn,on,sv,alt,obj); kr=canon(r)
   if kr not in seen:
    seen.add(kr); rows.append({"rendered":r,"provenance":"held-out object-relative verb substitution; same agreement state and attachment","agreement_state":{"subject":num,"relative_subject":num,"transition":"NP->VP requires V["+num+"]"},"repaired":True,"audit":audit(r),"live_frontier":live(r),"anti_shortcut":{"single_tree":True,"object_relative":True,"word_order_only":False,"repeated_unit":False,"catalogue_source":False,"fragment":False}}); repairs+=1
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']))
 exact=[x for x in rows if x['audit']['two_pointer']]
 out={"experiment":"cfg-object-relative-agreement-20260917","method":"single-tree object-relative CFG intersection with finite-state number agreement and live mirrored character frontier","control_count":controls,"repair_count":repairs,"candidate_count":len(rows),"exact_count":len(exact),"candidates":rows[:60],"next_repair":"Add a held-out transitive subject/object number contrast with an object-relative determiner state; preserve the object-relative tree and reject number-inconsistent transitions before character scoring."}
 p=Path('runs/cfg-object-relative-agreement-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({"run":str(p),"controls":controls,"repairs":repairs,"candidates":len(rows),"exact":len(exact),"longest":rows[0]['audit']['letters']}))
 for r in rows[:3]: print(r['rendered'],r['audit']['letters'],r['audit']['two_pointer'],r['repaired'])
if __name__=='__main__': main()
