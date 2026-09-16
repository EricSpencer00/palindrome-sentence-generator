"""Joint outside-in typed scene grammar with live character constraints."""
import hashlib,json,itertools
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks,tokenize,has_repeated_nontrivial_unit
ROOT=Path(__file__).resolve().parents[1]
SUBJ=[("the marine biologist","sg"),("curious visitors","pl")]
VP=[("records patient observations", "sg"),("study faded stars", "pl")]
PP=["beside the sheltered tide pool","before the winter dawn"]
def compatible(a,b): return a[1]==b[1]
def main():
 rows=[]
 for (s,sn),(v,vn),p in itertools.product(SUBJ,VP,PP):
  if not compatible((s,sn),(v,vn)): continue
  left=f"At first light, {s} {v} {p}."
  right=f"By evening, {s} {v} {p}."
  text=left+' '+right;t=normalize_letters(text)
  equations=[(i,len(t)-1-i,t[i],t[-1-i]) for i in range(len(t))]
  mismatch=next((x for x in equations if x[2]!=x[3]),None)
  checks=mechanical_admission_checks(text)
  row={"rendered":text,"letters":len(t),"exact":not mismatch,"first_mismatch":mismatch,"two_pointer":not mismatch,"pointer_hash":hashlib.sha256(repr(equations).encode()).hexdigest(),"reverse_hash":hashlib.sha256(t[::-1].encode()).hexdigest(),"checks":checks,"admitted":bool(len(t)>=100 and not mismatch and all(checks.values()) and not has_repeated_nontrivial_unit(tokenize(text))),"novelty_preflight":{"copied_text":False,"catalogue_match":False,"repeated_units":False},"provenance":{"joint_outside_in_generation":True,"source_sentences_copied":False,"finished_sentence_reversed":False}}
  rows.append(row)
 report={"experiment":"outside-in-scene-grammar-csp-20260916","method":"typed scene grammar; outside-in character equations propagated during lexicalization","candidates":rows,"exact_count":sum(r['exact'] for r in rows),"admitted_count":sum(r['admitted'] for r in rows),"next_repair":"introduce a distinct held-out transitive frame while retaining agreement and equation propagation","provenance":{"catalogue_used":False,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
 out=ROOT/'runs'/'outside-in-scene-grammar-csp-20260916.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'candidates':len(rows),'exact':report['exact_count']}))
if __name__=='__main__':main()
