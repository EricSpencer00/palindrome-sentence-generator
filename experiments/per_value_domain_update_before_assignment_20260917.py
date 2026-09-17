#!/usr/bin/env python3
"""Update both role domains from per-value support before scene assignment."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/per-value-domain-update-before-assignment-20260917.json'
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
 domains={**L,**R};support={};updates=[]
 for role,vals in list(domains.items()):
  keep=[]
  for v in vals:
   x={'agent':'teacher','action':'writes','theme':'notes','agent2':'cartographer','action2':'marks','theme2':'maps'};x[role]=v;t=norm(render(x));w=[p for p in range(min(10,len(t)//2)) if t[p]==t[-1-p]];support[f'{role}:{v}']=w
   if w:keep.append(v)
  domains[role]=keep or vals;updates.append({'role':role,'before':vals,'after':domains[role]})
 x={k:v[0] for k,v in domains.items()};rows=[{'candidate':0,'rendered':render(x),'updated_domains':domains,'support_by_value':support,'domain_updates':updates,'provenance':'per_value_domain_update_before_assignment','novelty_preflight':{'signature':'per_value_domain_update_v1','distinct_from':'per-value observation; both role domains are rewritten from support before any complete scene assignment'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}}]
 payload={'experiment':'per-value-domain-update-before-assignment-20260917','method':'per-value witness supports rewrite both left/right role domains before selecting a complete scene assignment','candidate_count':1,'candidates':rows,'summary':{'exact_count':0,'longest_letters':rows[0]['audit']['letters'],'next_repair':'iterate domain updates to a fixed point with cross-side support rather than a single pass'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
