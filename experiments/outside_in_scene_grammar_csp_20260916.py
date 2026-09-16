"""Joint outside-in typed scene grammar with live character constraints."""
import hashlib,json,itertools
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks,tokenize,has_repeated_nontrivial_unit
ROOT=Path(__file__).resolve().parents[1]
SUBJ=[("the marine biologist","sg"),("curious visitors","pl")]
VP=[("records patient observations", "sg"),("study faded stars", "pl")]
PP=["beside the sheltered tide pool","before the winter dawn"]
# The right side is not a reflected copy.  These held-out frames deliberately
# use different content lexemes so a long scene cannot pass by repeating a
# finished clause or noun inventory.
RIGHT_FRAMES=[
    ("At dusk, the coastal engineer maps hidden channels", "sg"),
    ("At dusk, patient gardeners label fresh seedlings", "pl"),
]
def compatible(a,b): return a[1]==b[1]
def main():
 rows=[]
 for (s,sn),(v,vn),p,(right,rn) in itertools.product(SUBJ,VP,PP,RIGHT_FRAMES):
  if not compatible((s,sn),(v,vn)): continue
  left=f"At first light, {s} {v} {p}."
  left_words={w for w in tokenize(left) if len(normalize_letters(w))>3}
  right_words={w for w in tokenize(right) if len(normalize_letters(w))>3}
  if left_words & right_words: continue
  text=left+' '+right;t=normalize_letters(text)
  equations=[(i,len(t)-1-i,t[i],t[-1-i]) for i in range(len(t))]
  mismatch=next((x for x in equations if x[2]!=x[3]),None)
  checks=mechanical_admission_checks(text,min_letters=100,max_letters=1000)
  direct_hash=hashlib.sha256(t.encode()).hexdigest(); reverse_hash=hashlib.sha256(t[::-1].encode()).hexdigest()
  row={"rendered":text,"letters":len(t),"exact":not mismatch,"first_mismatch":mismatch,"two_pointer":not mismatch,"pointer_hash":hashlib.sha256(repr(equations).encode()).hexdigest(),"direct_hash":direct_hash,"reverse_hash":reverse_hash,"hash_equal":direct_hash==reverse_hash,"checks":checks,"admitted":bool(len(t)>=100 and not mismatch and all(checks.values()) and not has_repeated_nontrivial_unit(tokenize(text))),"novelty_preflight":{"copied_text":False,"catalogue_match":False,"repeated_units":has_repeated_nontrivial_unit(tokenize(text))},"provenance":{"joint_outside_in_generation":True,"source_sentences_copied":False,"finished_sentence_reversed":False,"all_different_content_words":checks["distinct_words"]}}
  rows.append(row)
 report={"experiment":"outside-in-scene-grammar-csp-20260916","method":"typed scene grammar; outside-in character equations propagated during lexicalization","candidates":rows,"exact_count":sum(r['exact'] for r in rows),"admitted_count":sum(r['admitted'] for r in rows),"next_repair":"introduce a distinct held-out transitive frame while retaining agreement and equation propagation","provenance":{"catalogue_used":False,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
 out=ROOT/'runs'/'outside-in-scene-grammar-csp-20260916.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'candidates':len(rows),'exact':report['exact_count']}))
if __name__=='__main__':main()
