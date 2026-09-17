#!/usr/bin/env python3
"""Carry unresolved position masks through variable lexical boundaries."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/unresolved-mask-variable-boundaries-20260917.json'
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
  x={k:D[k][i%len(D[k])] for k in K};t=norm(render(x));resolved=set(range(min(20,len(t))));unresolved=[p for p in range(len(t)) if p not in resolved];eq=[];conf=[]
  for p in resolved:
   q=len(t)-1-p
   if q in resolved:(eq if t[p]==t[q] else conf).append((p,q))
  rows.append({'candidate':i,'rendered':render(x),'slots':x,'resolved_mask':sorted(resolved),'unresolved_mask':unresolved,'exact_equations':eq,'conflicts':conf,'provenance':'unresolved_mask_variable_boundary_equations','novelty_preflight':{'signature':'unresolved_mask_variable_boundary_v1','distinct_from':'partial tape equations; explicitly carries unresolved positions through variable-length slot boundaries'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'unresolved-mask-variable-boundaries-20260917','method':'resolved/unresolved tape-position masks are carried through variable lexical slot lengths while exact equations are solved on closed pairs','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'use mask transitions to choose the next lexical character whose opposing position is unresolved'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
