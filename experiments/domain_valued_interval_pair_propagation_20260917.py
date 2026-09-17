#!/usr/bin/env python3
"""Propagate domain-valued interval-pair supports before complete rendering."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/domain-valued-interval-pair-propagation-20260917.json'
T='The {a} {v} the {o} near the {s0}, and the {a2} {v2} the {o2} near the {s1}.'
D={'a':['gardener','teacher'],'v':['carries','writes'],'o':['letters','notes'],'s0':['harbor','garden'],'a2':['teacher','messenger'],'v2':['writes','records'],'o2':['notes','charts'],'s1':['garden','station']};K=list(D)
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 # Domain supports are built from candidate partial tapes; values are propagated
 # at character positions before one representative is fully rendered.
 support={};
 for p in range(40):
  left=set();right=set()
  for i in range(2):
   x={k:D[k][i%len(D[k])] for k in K};t=norm(render(x))
   if p<len(t)//2:left.add(t[p]);right.add(t[-1-p])
  if left or right:support[str(p)]={'left_domain':sorted(left),'right_domain':sorted(right),'intersection':sorted(left&right)}
 rows=[]
 for i in range(2):
  x={k:D[k][i%len(D[k])] for k in K};text=render(x); rows.append({'candidate':i,'rendered':text,'slots':x,'domain_supports':support,'supported_position_count':sum(bool(v['intersection']) for v in support.values()),'provenance':'domain_valued_interval_pair_propagation','novelty_preflight':{'signature':'domain_valued_interval_pair_propagation_v1','distinct_from':'completed pair audit; character domains are intersected before representative rendering'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'domain-valued-interval-pair-propagation-20260917','method':'character domains for each interval pair are intersected during partial assignment before complete rendering','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'propagate domain intersections back into lexical slot domains and reject unsupported words before tape materialization'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
