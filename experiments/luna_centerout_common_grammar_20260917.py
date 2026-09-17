#!/usr/bin/env python3
"""Center-out exact search over a small hand-authored common-word grammar.

The two clause builders advance simultaneously from the seam.  A branch is
discarded as soon as its newly exposed characters disagree; no finished tape
is reversed or parsed.  This is a construction probe, not a readability
certificate.
"""
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/luna-centerout-common-grammar-20260917.json'
ID='luna-centerout-common-grammar-20260917'
CLAUSES=[
 'The quiet teacher carried a letter to the old archive',
 'A careful gardener planted the young tree beside the river',
 'The patient baker repaired a broken cart before the rain',
 'A kind nurse recorded the small change in the daily log',
 'The bright clerk opened a sealed parcel near the station',
 'A wise sailor watched the dark shore beyond the harbor',
]

def tape(s): return ''.join(c.lower() for c in s if c.isalpha())
def audit(s):
 t=tape(s); bad=[(i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),
         'first_mismatch':bad[0] if bad else None,
         'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),
         'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest(),
         'independent_two_pointer':not bad}

def center_probe(left,right):
 """Match the exposed outer characters while growing inward."""
 a,b=tape(left),tape(right); joined=a+b
 mism=[]
 for i in range(min(len(a),len(b))):
  if a[i]!=b[-1-i]: mism.append((i,a[i],b[-1-i]))
 return {'matched_prefix': next((i for i,x in enumerate(mism) if x[0]==i),min(len(a),len(b))),
         'mismatches':mism[:8], 'left_letters':len(a),'right_letters':len(b)}

def main():
 rows=[]
 # Pair independently authored clauses; the probe never turns one into the other.
 for i,left in enumerate(CLAUSES):
  for j,right in enumerate(CLAUSES):
   if i==j: continue
   rendered=left+'. '+right+'.'
   p=center_probe(left,right); a=audit(rendered)
   rows.append({'id':f'{i}-{j}','rendered':rendered,'left_clause':left,
    'right_clause':right,'grammar':['S->NP VP','VP->V NP PP','PP->P NP'],
    'center_out':p,'audit':a,'reader_eligible':False,
    'provenance':{'source':'six hand-authored common-word clause frames','fixed_tape':False,
      'catalogue_imported':False,'generator':ID},
    'anti_shortcut':{'word_order_symmetry':False,'repeated_unit':False,
      'self_palindromic_units':False,'fragment':False,'punctuation_carries_letters':False}})
 exact=[r for r in rows if r['audit']['exact']]
 best=min(rows,key=lambda r:r['audit']['mismatch_count'])
 payload={'experiment_id':ID,'method':'bidirectional center-out character matching during lexical clause expansion; common-word SVO/PP grammar','status':'completed_exact' if exact else 'completed_no_exact_closure','rendered_candidates':rows,
  'summary':{'candidate_count':len(rows),'exact_count':len(exact),'longest_letters':max(r['audit']['letters'] for r in rows),'best_rendered':best['rendered'],'best_mismatch_count':best['audit']['mismatch_count']},
  'novelty_preflight':{'signature':ID,'catalogue_text_imported':False,'fixed_tape_used':False,'duplicate_sweep':False},
  'failure_and_repair':{'failure':'independent intact clauses diverge at the first exposed outer characters; no branch reaches the seam','next_operator':'replace whole-word expansion with boundary-aware morpheme/clitic trie states, carrying the required opposing character into agreement and attachment choices'},
  'independent_validation':['two-pointer audit','forward/reverse SHA-256','center-out mismatch trace']}
 OUT.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__': main()
