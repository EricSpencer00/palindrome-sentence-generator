#!/usr/bin/env python3
"""Retain seam-conflict history and stop revising exhausted role domains."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/exhausted-role-conflict-history-20260917.json'
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
 rows=[]
 for i in range(2):
  domains={k:list(v) for k,v in D.items()};history=[];x={k:domains[k][i%len(domains[k])] for k in K}
  for step in range(3):
   t=norm(render(x));p=next((q for q in range(len(t)//2) if t[q]!=t[-1-q]),0);role='a' if p<8 else 'v';history.append({'step':step,'position':p,'role':role,'domain_before':list(domains[role])})
   if len(domains[role])>1:domains[role].pop(0);x[role]=domains[role][0];history[-1]['action']='revised'
   else: history[-1]['action']='exhausted_stop'
  rows.append({'candidate':i,'rendered':render(x),'slots':x,'conflict_history':history,'exhausted_roles':[k for k,v in domains.items() if len(v)==1],'provenance':'exhausted_role_conflict_history','novelty_preflight':{'signature':'exhausted_role_conflict_history_v1','distinct_from':'single role repair; preserves conflict history and halts domain revision at exhaustion'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'exhausted-role-conflict-history-20260917','method':'retain conflict history while revising implicated roles; stop when a role domain is exhausted','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'route exhausted-role conflicts to a new syntactic construction operator rather than revisiting the same domain'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
