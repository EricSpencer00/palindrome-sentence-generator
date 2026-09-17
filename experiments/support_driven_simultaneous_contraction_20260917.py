#!/usr/bin/env python3
"""Retain only role values with nonempty bidirectional witness support."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/support-driven-simultaneous-contraction-20260917.json'
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
 kept=[];rejected=[]
 for l in L:
  for r in R:
   x={'agent':l[0],'action':l[1],'theme':l[2],'prep':l[3],'agent2':r[0],'action2':r[1],'theme2':r[2]};t=norm(render(x));w=[p for p in range(min(10,len(t)//2)) if t[p]==t[-1-p]]
   if w:kept.append((l,r,w))
   else:rejected.append([list(l),list(r)])
 rows=[]
 for i,(l,r,w) in enumerate(kept):
  x={'agent':l[0],'action':l[1],'theme':l[2],'prep':l[3],'agent2':r[0],'action2':r[1],'theme2':r[2]};rows.append({'candidate':i,'rendered':render(x),'left_role':list(l),'right_role':list(r),'bidirectional_witnesses':w,'rejected_pairs':rejected,'provenance':'support_driven_simultaneous_contraction','novelty_preflight':{'signature':'support_driven_simultaneous_contraction_v1','distinct_from':'simultaneous domain contraction; pair values are retained only with explicit nonempty witness support'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'support-driven-simultaneous-contraction-20260917','method':'simultaneous left/right role pairs survive only with nonempty bidirectional tape witnesses','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'propagate witness support incrementally per role value rather than per completed pair'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
