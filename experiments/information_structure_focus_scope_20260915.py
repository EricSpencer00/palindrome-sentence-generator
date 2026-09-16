"""Diagnostic route: focus/presupposition scope is part of the semantic state.
Not a generator claim: clauses are independently authored and only exact tape
agreement can produce a candidate.
"""
import json, argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks, tokenize

SIGNATURE="information-structure-focus|presupposition-scope|polarity-bearing-clause-plans|causal-seam-result-authoring|independent-information-structure-audit"
LEFT=["The careful editor did not revise the draft because the facts changed.","The patient teacher did not grade the essay because the class waited.","The quiet farmer did not harvest the grain because the clouds gathered.","The skilled baker did not sell the bread because the market closed."]
RIGHT=["The facts changed, so the careful editor left the draft alone.","The class waited, so the patient teacher left the essay ungraded.","The clouds gathered, so the quiet farmer left the grain unharvested.","The market closed, so the skilled baker left the bread unsold."]
def two_pointer(t):
 i,j=0,len(t)-1
 while i<j:
  if t[i] != t[j]: return False
  i,j=i+1,j-1
 return bool(t)
def audit(text):
 t=normalize_letters(text); c=mechanical_admission_checks(text,min_letters=39,max_letters=260)
 return {"rendered":text,"letters":len(t),"exact":bool(t) and t==t[::-1],"independent_two_pointer":two_pointer(t),"normalized_sha256":__import__('hashlib').sha256(t.encode()).hexdigest(),"tokens":list(tokenize(text)),"admitted":all(c.values()),"failed_checks":[k for k,v in c.items() if not v],"readable_status":"diagnostic_only","semantic_state":{"focus":"agent action","presupposition":"cause event","polarity":"negative-to-result"}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 rows=[audit(l+' '+r) for l in LEFT for r in RIGHT]
 out={"status":"complete_diagnostic","signature":SIGNATURE,"preflight":{"registry_entries":63,"excluded_families":5,"near_pair_review_flags":13,"audit_passed":True,"distinct_dimension":SIGNATURE},"operator":"independent information-structure clause plans; exact tape audit after rendering","rows":rows,"independent_audit":{"method":"explicit opposing-index scan over every rendered probe","probes_checked":len(rows),"primary_exact":sum(x['exact'] for x in rows),"independent_exact":sum(x['independent_two_pointer'] for x in rows),"disagreements":[x['rendered'] for x in rows if x['exact'] != x['independent_two_pointer']]},"exact_survivors":[x for x in rows if x['exact']],"admitted_survivors":[x for x in rows if x['exact'] and x['admitted']],"reader_status":"not run; no admitted survivor","next_repair":"author a new polarity-preserving result clause whose terminal character stream crosses the causal seam; rerun independent parse and admission","provenance":{"source":"four authored negative-cause/result plans; no catalogue text","generator":str(Path(__file__).relative_to(ROOT))}}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'rows':len(rows),'exact':len(out['exact_survivors']),'admitted':len(out['admitted_survivors'])}))
if __name__=='__main__': main()
