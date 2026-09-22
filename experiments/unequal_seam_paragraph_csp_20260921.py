"""Unequal semantic seam CSP over the 120-letter Nora/Aron paragraph tape."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs'/'unequal-seam-paragraph-csp-20260921.json'
sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks
UNITS=["Nora saw lager.","Nora saw desserts.","Nora saw trams.","Nora saw guns.","Nora saw war.",
       "Raw was Aron.","Snug was Aron.","Smart was Aron.","Stressed was Aron.","Regal was Aron."]
def norm(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=norm(s); mm=[{'offset':i,'left':t[i],'right':t[-1-i]} for i in range(len(t)//2) if t[i]!=t[-1-i]]
 f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'letters':len(t),'two_pointer_exact':bool(t) and not mm,'first_mismatches':mm[:4], 'forward_sha256':f,'reverse_sha256':r,'sha_equal':f==r}
def span(units): return ' '.join(units)
def run():
 # Deliberately unequal semantic partition: complete clauses cross the old
 # paragraph seam, while the physical tape still closes through word edges.
 parts=[UNITS[:4],UNITS[4:5],UNITS[5:9],UNITS[9:10]]
 labels=['A','B','B_prime','A_prime']; spans={k:span(v) for k,v in zip(labels,parts)}
 rendered=' '.join(UNITS); t=audit(rendered)
 pair_checks=[]
 for left,right in [('A','A_prime'),('B','B_prime')]:
  l,r=norm(spans[left]),norm(spans[right]); pair_checks.append({'left':left,'right':right,'left_letters':len(l),'right_letters':len(r),'exact_reverse':l==r[::-1],'proper_palindromic_center': l==l[::-1] if left=='B' else False})
 local_checks={'whole_tape_exact':t['two_pointer_exact'],'semantic_spans_complete_clauses':all(s.endswith('.') for s in spans.values()),'unequal_A_A_prime':len(norm(spans['A']))!=len(norm(spans['A_prime'])),'unequal_B_B_prime':len(norm(spans['B']))!=len(norm(spans['B_prime'])),'center_not_proper_palindrome':norm(spans['B'])!=norm(spans['B'])[::-1]}
 central=mechanical_admission_checks(rendered,min_letters=30,max_letters=2000)
 candidate={'rendered':rendered,'semantic_spans':spans,'pair_checks':pair_checks,'audit':t,'local_seam_checks':local_checks,'central_mechanical_admission':{'checks':central,'admitted':all(central.values()),'blocking_failures':sorted(k for k,v in central.items() if not v)},'provenance':{'construction':'unequal semantic labels over the pre-existing mirrored Nora/Aron control','whole_tape_solved_jointly':False,'finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False,'word_order_symmetry':True,'repeated_phrase_scaffold':True,'distinct_units':len(set(UNITS))==len(UNITS),'self_palindromic_units':[u for u in UNITS if norm(u)==norm(u)[::-1]]}}
 assert all(local_checks.values())
 assert not candidate['central_mechanical_admission']['admitted']
 assert not central['not_word_order_symmetry']
 return {'experiment_id':'unequal-seam-paragraph-csp-20260921','method':'audit unequal semantic labels against the central anti-shortcut gate','candidate':candidate,'stats':{'candidate_count':1,'letters':t['letters'],'exact_controls':1,'mechanically_admitted':0,'pair_reverse_exact_count':sum(x['exact_reverse'] for x in pair_checks)},'novelty_preflight':{'status':'failed_after_central_audit','signature':'unequal-semantic-seams|cross-boundary-word-ledger|nora-aron','reason':'semantic repartitioning does not change the pre-existing word-symmetric tape'},'independent_validation':['two-pointer full tape','forward/reverse SHA-256','central mechanical_admission_checks'],'status':'rejected exact control; no reader candidate','next_operator':'generate a fresh varied-frame tape with cross-word seams; do not relabel the mirrored Nora/Aron control'}
if __name__=='__main__':
 d=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(d,indent=2)+'\n'); print(json.dumps(d['stats'],sort_keys=True))
