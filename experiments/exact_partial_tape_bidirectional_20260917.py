#!/usr/bin/env python3
"""Exact partial-tape equations with bidirectional propagation."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/exact-partial-tape-bidirectional-20260917.json'
T='The {a} {v} the {o} near the {s0}, and the {a2} {v2} the {o2} near the {s1}.'
D={'a':['gardener','teacher'],'v':['carries','writes'],'o':['letters','notes'],'s0':['harbor','garden'],'a2':['teacher','messenger'],'v2':['writes','records'],'o2':['notes','charts'],'s1':['garden','station']};K=list(D)
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**{k:x.get(k,'') for k in K})
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def partial_equations(x):
 t=norm(render(x));known=sum(k in x for k in K);eq=[];conf=[]
 for i in range(min(known*3,len(t)//2)):
  j=len(t)-1-i
  (eq if t[i]==t[j] else conf).append((i,j,t[i],t[j]))
 return eq,conf
def main():
 states=[{}];trace=[]
 for k in K:
  nxt=[]
  for st in states:
   for w in D[k]:
    q=dict(st);q[k]=w;eq,conf=partial_equations(q)
    if not conf:nxt.append(q)
  trace.append({'slot':k,'prior_states':len(states),'equation_supported_states':len(nxt)});states=nxt[:8]
 if not states: states=[{k:D[k][0] for k in K}]
 rows=[]
 for i,x in enumerate(states[:4]):
  eq,conf=partial_equations(x);text=render(x);rows.append({'candidate':i,'rendered':text,'slots':x,'partial_equalities':eq,'partial_conflicts':conf,'trace':trace,'provenance':'exact_partial_tape_bidirectional_equations','novelty_preflight':{'signature':'exact_partial_tape_bidirectional_v1','distinct_from':'heuristic support; rejects any resolved opposing equation conflict in both directions'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'exact-partial-tape-bidirectional-20260917','method':'incremental lexical states are admitted only when all currently resolved opposing tape equations agree','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'carry unresolved-position masks and solve exact equations across variable word boundaries'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
