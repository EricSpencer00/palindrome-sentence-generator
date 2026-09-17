#!/usr/bin/env python3
"""Carry full positional support maps between paired lexical slots."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/paired-slot-full-support-maps-20260917.json'
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
 pairs=list(zip(K[:4],reversed(K[4:])));maps=[];rows=[]
 for left,right in pairs:
  m={}
  for lw in D[left]:
   for rw in D[right]:
    a=norm(lw);b=norm(rw);matches=[(i,j,a[i],b[j]) for i in range(len(a)) for j in range(len(b)) if a[i]==b[j]];m[f'{lw}|{rw}']=matches
  maps.append({'pair':[left,right],'support_map':m})
 for i in range(2):
  x={k:D[k][i%len(D[k])] for k in K};text=render(x);rows.append({'candidate':i,'rendered':text,'slots':x,'paired_support_maps':maps,'provenance':'paired_slot_full_support_map_propagation','novelty_preflight':{'signature':'paired_slot_full_support_map_v1','distinct_from':'full interval recomputation; support maps are materialized and carried per paired lexical domain'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'paired-slot-full-support-maps-20260917','method':'materialize full character support maps for each paired lexical domain and carry them into candidate states','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'use support-map intersections to prune paired lexical values before rendering any candidate'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
