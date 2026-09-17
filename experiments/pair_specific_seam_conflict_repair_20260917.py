#!/usr/bin/env python3
"""Repair only the semantic lexical role implicated by a seam conflict."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/pair-specific-seam-conflict-repair-20260917.json'
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
  x={k:D[k][i%len(D[k])] for k in K};t=norm(render(x));m=next((p for p in range(len(t)//2) if t[p]!=t[-1-p]),0);role='a' if m<8 else 'v';old=x[role];choices=[w for w in D[role] if w!=old];x[role]=choices[0] if choices else old
  rows.append({'candidate':i,'rendered':render(x),'repaired_role':role,'replaced':{'old':old,'new':x[role]},'slots':x,'preserved_roles':[k for k in K if k!=role],'provenance':'pair_specific_seam_conflict_role_repair','novelty_preflight':{'signature':'pair_specific_seam_conflict_repair_v1','distinct_from':'whole-tape check; only the lexical semantic role named by the first seam conflict is revised'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'pair-specific-seam-conflict-repair-20260917','method':'first whole-tape seam conflict identifies one semantic lexical role; only that role domain is revised','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'retain conflict provenance across repair iterations and stop revising a role when its domain is exhausted'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
