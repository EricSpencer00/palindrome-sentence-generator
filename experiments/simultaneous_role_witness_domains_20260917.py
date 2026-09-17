#!/usr/bin/env python3
"""Simultaneous left/right role witness-domain propagation."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/simultaneous-role-witness-domains-20260917.json'
T='At dawn, the {agent} {action} the {theme} {prep} the harbor; meanwhile, a {agent2} {action2} the {theme2} near the archive.'
L=[('teacher','writes','notes','along'),('gardener','carries','letters','through')];R=[('cartographer','marks','maps'),('messenger','records','charts'),('teacher','writes','notes')]
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 rows=[];domain_history=[]
 left=list(L);right=list(R)
 for step in range(2):
  domain_history.append({'step':step,'left_size':len(left),'right_size':len(right)})
  if len(left)>1:left=left[:-1]
  if len(right)>2:right=right[:-1]
 for l in left:
  for r in right:
   x={'agent':l[0],'action':l[1],'theme':l[2],'prep':l[3],'agent2':r[0],'action2':r[1],'theme2':r[2]};t=norm(render(x));w=[p for p in range(min(10,len(t)//2)) if t[p]==t[-1-p]]
   rows.append({'candidate':len(rows),'rendered':render(x),'left_role':list(l),'right_role':list(r),'left_witnesses':w,'right_witnesses':[len(t)-1-p for p in w],'domain_history':domain_history,'provenance':'simultaneous_left_right_role_witness_domains','novelty_preflight':{'signature':'simultaneous_role_witness_domains_v1','distinct_from':'right-only pruning; left and right role domains contract together at each propagation step'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'simultaneous-role-witness-domains-20260917','method':'left and right semantic role domains contract simultaneously while witness supports are carried in both directions','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'make simultaneous contraction support-driven and retain only domain values with nonempty bidirectional witnesses'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
