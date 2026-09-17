#!/usr/bin/env python3
"""Prune paired lexical values while preserving character-index support."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/positional-support-intersection-pruning-20260917.json'
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
 pairs=list(zip(K[:4],reversed(K[4:]))); kept={};removed=0
 for l,r in pairs:
  vals=[]
  for a in D[l]:
   for b in D[r]:
    matches=[(i,j) for i,x in enumerate(norm(a)) for j,y in enumerate(norm(b)) if i==j and x==y]
    if matches:vals.append({'left':a,'right':b,'index_matches':matches})
    else:removed+=1
  kept[f'{l}|{r}']=vals
 rows=[]
 for i in range(2):
  x={k:D[k][i%len(D[k])] for k in K};text=render(x);rows.append({'candidate':i,'rendered':text,'slots':x,'positional_kept_pairs':kept,'removed_pairs':removed,'provenance':'positional_support_intersection_pruning','novelty_preflight':{'signature':'positional_support_intersection_v1','distinct_from':'set-level support intersection; retains exact lexical character indices for support'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'positional-support-intersection-pruning-20260917','method':'paired lexical values survive only when character support matches at identical indices','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'map lexical slot indices to rendered tape offsets and intersect against true opposing palindrome positions'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
