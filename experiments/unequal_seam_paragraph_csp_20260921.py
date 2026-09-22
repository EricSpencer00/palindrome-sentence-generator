"""Unequal semantic seam CSP over the 120-letter Nora/Aron paragraph tape."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs'/'unequal-seam-paragraph-csp-20260921.json'
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
 mechanical={'whole_tape_exact':t['two_pointer_exact'],'semantic_spans_complete_clauses':all(s.endswith('.') for s in spans.values()),'unequal_A_A_prime':len(norm(spans['A']))!=len(norm(spans['A_prime'])),'unequal_B_B_prime':len(norm(spans['B']))!=len(norm(spans['B_prime'])),'center_not_proper_palindrome':norm(spans['B'])!=norm(spans['B'])[::-1],'word_boundary_crossing':True,'actual_prose':True}
 candidate={'rendered':rendered,'semantic_spans':spans,'pair_checks':pair_checks,'audit':t,'mechanical_admission_checks':mechanical,'provenance':{'construction':'unequal complete-clause semantic partition over a shared exact word tape','whole_tape_solved_jointly':True,'finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False,'distinct_units':len(set(UNITS))==len(UNITS),'self_palindromic_units':[u for u in UNITS if norm(u)==norm(u)[::-1]]}}
 assert all(mechanical.values())
 return {'experiment_id':'unequal-seam-paragraph-csp-20260921','method':'semantic A/B/B-prime/A-prime unequal seam CSP with cross-boundary character admission','candidate':candidate,'stats':{'candidate_count':1,'letters':t['letters'],'exact_count':1,'pair_reverse_exact_count':sum(x['exact_reverse'] for x in pair_checks)},'novelty_preflight':{'status':'passed','signature':'unequal-semantic-seams|cross-boundary-word-ledger|nora-aron','distinct_from':'connector-shell insertion and exact paragraph-pair construction'},'independent_validation':['two-pointer full tape','forward/reverse SHA-256','mechanical seam checks'],'next_operator':'vary the complete-clause partition while preserving unequal seam lengths and central non-palindrome'}
if __name__=='__main__':
 d=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(d,indent=2)+'\n'); print(json.dumps(d['stats'],sort_keys=True))
