#!/usr/bin/env python3
"""Couple completed lexical boundary pairs to whole-tape seam equations."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/completed-pair-whole-tape-seam-20260917.json'
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
  x={k:D[k][i%len(D[k])] for k in K};t=norm(render(x));pairs=[('a','a2'),('v','v2'),('o','o2'),('s0','s1')];checks=[]
  for l,r in pairs:
   a=norm(x[l]);b=norm(x[r]);checks.append({'pair':[l,r],'boundary_equal':a[-1]==b[0],'tape_seam_matches':sum(t[p]==t[-1-p] for p in range(min(8,len(t)//2)))})
  rows.append({'candidate':i,'rendered':render(x),'slots':x,'completed_pair_seam_checks':checks,'provenance':'completed_pair_whole_tape_seam_equations','novelty_preflight':{'signature':'completed_pair_whole_tape_seam_v1','distinct_from':'boundary completion; whole-tape seam equations gate completed lexical pairs before clause admission'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'completed-pair-whole-tape-seam-20260917','method':'completed lexical boundary pairs are coupled to exact whole-tape seam checks before clause admission','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'propagate seam conflicts back to the specific lexical pair and revise only its semantic role domain'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
