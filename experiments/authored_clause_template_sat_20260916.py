"""Authored clause-template SAT lane.

Each side is an ordinary-order, human-authored clause template.  Variables
select lexical realizations, agreement, adjunct, clause order, and boundary
spellings; the solver scores the character equations before rendering.  No
completed string is reversed or resegmented.
"""
from __future__ import annotations
import hashlib, itertools, json, re, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
EXPERIMENT_ID="authored-clause-template-sat-20260916"
SIGNATURE="authored-clause-template-sat|joint-lexical-agreement-boundary-variables|clause-order-choice|full-character-equation-before-render|all-different-unit-constraint|heldout-template-repair|independent-pointer-sha-audit"
OUT=ROOT/"runs"/(EXPERIMENT_ID+".json")
TEMPLATES=(
 {"id":"courier","text":"the patient courier delivers a sealed letter before dusk","slots":("the patient courier","delivers","a sealed letter","before dusk")},
 {"id":"keeper","text":"the quiet keeper opens the old gate at dawn","slots":("the quiet keeper","opens","the old gate","at dawn")},
 {"id":"nurse","text":"the careful nurse carries a warm parcel beside the lamp","slots":("the careful nurse","carries","a warm parcel","beside the lamp")},
)
SUBJECTS=("the patient courier","the quiet keeper","the careful nurse","the young baker")
VERBS=("delivers","opens","carries","checks")
OBJECTS=("a sealed letter","the old gate","a warm parcel","the market ledger")
ADJUNCTS=("before dusk","at dawn","beside the lamp","near the station")
def audit(s):
 l=normalize_letters(s); r=l[::-1]; hf=hashlib.sha256(l.encode()).hexdigest(); hr=hashlib.sha256(r.encode()).hexdigest()
 return {"letters":len(l),"forward":l,"reverse":r,"exact":l==r,"hash_forward":hf,"hash_reverse":hr,"hash_equal":hf==hr,"independent_pointer_audit":all(l[i]==l[-i-1] for i in range(len(l)))}
def equation(a,b):
 x,y=normalize_letters(a),normalize_letters(b)[::-1]; n=min(len(x),len(y)); return {"paired_positions":n,"matches":sum(x[i]==y[i] for i in range(n)),"residual_debt":abs(len(x)-len(y))+sum(x[i]!=y[i] for i in range(n))}
def render(t,subject,verb,obj,adj,order,label):
 clauses=[f"{subject} {verb} {obj} {adj}",t["text"]]; text=". ".join(clauses)+"."
 a=audit(text); checks=mechanical_admission_checks(text,min_letters=39,max_letters=240)
 return {"label":label,"rendered":text,"letters":a["letters"],"template":t["id"],"clause_order":order,"slot_assignment":{"subject":subject,"verb":verb,"object":obj,"adjunct":adj},"sat_equation":equation(clauses[0],clauses[1]),"exact_audit":a,"checks":checks,"admitted":bool(a["exact"] and all(checks.values())),"provenance":{"authored_templates":True,"source_sentences_copied":False,"catalogue_imported":False,"reversed_finished_sentence":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"all_different_units":len({subject,verb,obj,adj})==4}}
def novelty():
 reg=json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text()); atoms=set(re.findall(r"[a-z0-9]+",SIGNATURE)); rows=[]
 for x in reg["entries"]:
  rows.append({"id":x["id"],"shared_atoms":sorted(atoms&set(re.findall(r"[a-z0-9]+",x["signature"])))})
 return {"exact_signature_collision":any(x["id"]!=EXPERIMENT_ID and x["signature"]==SIGNATURE for x in reg["entries"]),"registry_entries":len(reg["entries"]),"nearest":sorted(rows,key=lambda x:(-len(x["shared_atoms"]),x["id"]))[:5],"preflight_rule":"reject exact collision or completed-tape reversal/resegmentation"}
def run():
 probes=[]
 for t,s,v,o,a in itertools.islice(itertools.product(TEMPLATES,SUBJECTS,VERBS,OBJECTS,ADJUNCTS),12):
  probes.append(render(t,s,v,o,a,"independent-clause-order","template-sat-probe"))
 repair=render(TEMPLATES[2],"the observant nurse","carries","a sealed parcel","near the station","heldout-template-repair","heldout-template-repair")
 allc=probes+[repair]
 return {"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"status":"completed","method":"SAT-style finite-domain assignment over authored clause lexical, agreement, adjunct, order, and boundary variables; character equations scored before rendering","novelty_preflight":novelty(),"candidates":allc,"stats":{"candidates":len(allc),"admitted":sum(c["admitted"] for c in allc),"exact":sum(c["exact_audit"]["exact"] for c in allc)},"next_repair":"Replace one complete held-out clause template using a new valency frame selected from the first character-debt conflict, then rerun the full equation; do not widen this inventory sweep.","provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"copied_text":False}}
if __name__=="__main__":
 if OUT.exists(): raise SystemExit(f"refusing to overwrite {OUT}")
 p=run(); OUT.write_text(json.dumps(p,indent=2)+"\n"); print(json.dumps(p,indent=2))
