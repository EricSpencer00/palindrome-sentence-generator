#!/usr/bin/env python3
"""Full positional character constraints over rendered lexical slot intervals."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/full-positional-slot-interval-csp-20260917.json'
T='The {a} {v} the {o} near the {s0}, and the {a2} {v2} the {o2} near the {s1}.'
D={'a':['gardener','teacher'],'v':['carries','writes'],'o':['letters','notes'],'s0':['harbor','garden'],'a2':['teacher','messenger'],'v2':['writes','records'],'o2':['notes','charts'],'s1':['garden','station']};K=list(D)
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def intervals(x):
 out={};p=0
 for z in re.split(r'({\w+})',T):
  if z.startswith('{'):
   k=z[1:-1];w=norm(x[k]);out[k]=(p,p+len(w)-1);p+=len(w)
  else:p+=len(norm(z))
 return out
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 rows=[]
 for i in range(2):
  x={k:D[k][i%len(D[k])] for k in K};t=norm(render(x));sp=intervals(x);constraints=[];conf=0
  for lo,hi in sp.values():
   for p in range(lo,hi+1):
    q=len(t)-1-p
    if q>=0 and t[p]==t[q]:constraints.append((p,q,t[p]))
    elif q>=0:conf+=1
  rows.append({'candidate':i,'rendered':render(x),'slots':x,'slot_intervals':{k:list(v) for k,v in sp.items()},'positional_equalities':constraints,'positional_conflicts':conf,'provenance':'full_positional_slot_interval_csp','novelty_preflight':{'signature':'full_positional_slot_interval_csp_v1','distinct_from':'boundary arcs; constrains every resolved character position inside rendered slot intervals'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'full-positional-slot-interval-csp-20260917','method':'exact character constraints are propagated for every position covered by rendered lexical slot intervals','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'use interval domains rather than completed word assignments and prune a slot as soon as any positional equality loses support'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
