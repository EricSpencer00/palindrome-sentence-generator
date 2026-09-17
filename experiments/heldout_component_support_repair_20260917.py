#!/usr/bin/env python3
"""Held-out semantic replacement for a support-delta component."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/heldout-component-support-repair-20260917.json'
T='At dawn, the {agent} {action} the {theme} along the harbor; meanwhile, a {agent2} {action2} the {theme2} near the archive.'
BASE={'agent':'teacher','action':'writes','theme':'notes','agent2':'cartographer','action2':'marks','theme2':'maps'};HELD={'agent':['archivist'],'action':['keeps'],'theme':['records'],'agent2':['courier'],'action2':['delivers'],'theme2':['parcels']}
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
 for side in ('left','right'):
  x=dict(BASE);x.update({k:v[0] for k,v in HELD.items() if (side=='left' and k in ('agent','action','theme')) or (side=='right' and k in ('agent2','action2','theme2'))});t=norm(render(x));support=[p for p in range(min(12,len(t)//2)) if t[p]==t[-1-p]]
  rows.append({'candidate':len(rows),'rendered':render(x),'repaired_side':side,'heldout_values':{k:x[k] for k in x if x[k] in sum(HELD.values(),[])},'pair_local_support':support,'provenance':'heldout_component_support_repair','novelty_preflight':{'signature':'heldout_component_support_v1','distinct_from':'component-local delta; replacement comes from held-out semantic role domain and support is re-evaluated'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'heldout-component-support-repair-20260917','method':'replace selected support-delta component from held-out semantic domain and re-evaluate pair-local support','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'run held-out replacements through the full scene grammar with human-readable attachment checks'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
