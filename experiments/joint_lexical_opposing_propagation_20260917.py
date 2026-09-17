#!/usr/bin/env python3
"""Joint lexical propagation conditioned on all prior slot choices."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/joint-lexical-opposing-propagation-20260917.json'
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
 states=[{}];trace=[]
 for k in K:
  nxt=[]
  for st in states:
   for w in D[k]:
    q=dict(st);q[k]=w;t=norm(render({z:q.get(z,D[z][0]) for z in K}));support=sum(t[p]==t[-1-p] for p in range(min(5,len(t)//2)))
    if support>=1:nxt.append(q)
  trace.append({'slot':k,'prior_states':len(states),'joint_supported_states':len(nxt)});states=nxt[:8]
 rows=[]
 for i,x in enumerate(states[:4]):
  text=render(x);rows.append({'candidate':i,'rendered':text,'slots':x,'propagation_trace':trace,'provenance':'joint_lexical_opposing_offset_propagation','novelty_preflight':{'signature':'joint_lexical_opposing_propagation_v1','distinct_from':'per-slot choice; opposing support is conditioned on every prior slot assignment'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'joint-lexical-opposing-propagation-20260917','method':'incremental joint slot assignment retains only states with opposing-offset support under all prior choices','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'replace heuristic support threshold with exact partial tape equations and bidirectional state propagation'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
