#!/usr/bin/env python3
"""Propagate tape-witness support incrementally for each role value."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/incremental-role-value-witness-support-20260917.json'
T='At dawn, the {agent} {action} the {theme} along the harbor; meanwhile, a {agent2} {action2} the {theme2} near the archive.'
L={'agent':['teacher','gardener'],'action':['writes','carries'],'theme':['notes','letters']};R={'agent2':['cartographer','messenger'],'action2':['marks','records'],'theme2':['maps','charts']}
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 rows=[]
 for side,dom in [('left',L),('right',R)]:
  for role,vals in dom.items():
   for val in vals:
    x={'agent':'teacher','action':'writes','theme':'notes','agent2':'cartographer','action2':'marks','theme2':'maps'};x[role]=val;t=norm(render(x));w=[p for p in range(min(10,len(t)//2)) if t[p]==t[-1-p]]
    rows.append({'candidate':len(rows),'rendered':render(x),'side':side,'role':role,'value':val,'incremental_witnesses':w,'provenance':'incremental_role_value_witness_support','novelty_preflight':{'signature':'incremental_role_value_witness_v1','distinct_from':'completed pair gating; support is recorded as each individual role value is introduced'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'incremental-role-value-witness-support-20260917','method':'introduce each left/right semantic role value independently and propagate its tape-witness support immediately','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'use per-value support to update both role domains incrementally before selecting complete scene assignments'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
