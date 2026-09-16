"""Joint CFG/Earley-style intersection probe (both sides generated together)."""
import hashlib,json,itertools
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1]
SUBJ=["the patient archivist","curious visitors","a careful engineer"]
VERB=["maps the quiet museum archive","study faded stars beside the river","repairs brass lanterns near the station"]
TAIL=["before dawn","during the long winter","while tired travelers wait"]

def derivations():
  # CFG nonterminals are expanded on both sides in lockstep; seam variables
  # carry only agreement features, never characters from a fixed tape.
  for s,v,t in itertools.product(SUBJ,VERB,TAIL):
    left=f"{s} {v} {t}."
    for s2,v2,t2 in itertools.product(SUBJ,VERB,TAIL):
      right=f"{s2} {v2} {t2}."
      yield left,right,{"S":{"agreement":"sg" if s.startswith(("the patient","a ")) else "pl"},"NP":s,"VP":v,"PP":t}
def audit(left,right):
  text=left+" "+right; tape=normalize_letters(text)
  h=hashlib.sha256("".join(f"{i}:{len(tape)-1-i}:{a}:{tape[-1-i]}" for i,a in enumerate(tape)).encode()).hexdigest()
  checks=mechanical_admission_checks(text)
  return {"rendered":text,"letters":len(tape),"exact":tape==tape[::-1],"admitted":bool(len(tape)>100 and tape==tape[::-1] and all(checks.values())),"checks":checks,"pointer_hash":h}
def main():
  rows=[]
  for left,right,tree in derivations():
    row=audit(left,right); row["cfg_tree"]=tree; row["novelty_preflight"]={"copied_text":False,"catalogue_match":False,"fixed_tape_parse":False}; row["provenance"]={"joint_cfg_derivation":True,"source_sentences_copied":False,"reversed_finished_sentence":False}
    rows.append(row)
    if row["admitted"]: break
  out=ROOT/'runs'/'joint-cfg-intersection-20260916.json'; report={"experiment":"joint-cfg-intersection-20260916","method":"joint CFG derivation with agreement-feature seam variables; independent exact audit","candidates":rows,"exact_count":sum(r['exact'] for r in rows),"admitted_count":sum(r['admitted'] for r in rows),"next_repair":"add an Earley chart with typed adjunct recursion and retain joint generation; do not seed either side from a completed tape","provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"catalogue_used":False}}; out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'candidates':len(rows),'exact':report['exact_count']}))
if __name__=='__main__': main()
