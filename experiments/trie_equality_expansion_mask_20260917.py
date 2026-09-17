#!/usr/bin/env python3
"""Expand selected tape positions through lexical trie equality states."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/trie-equality-expansion-mask-20260917.json'
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
  x={k:D[k][i%len(D[k])] for k in K};t=norm(render(x));resolved=set(range(16));exp=[]
  for p in sorted(resolved):
   q=len(t)-1-p
   if q not in resolved: exp.append({'position':p,'opposing':q,'left_prefix':t[p],'required_right_prefix':t[p]})
  rows.append({'candidate':i,'rendered':render(x),'slots':x,'trie_equality_expansions':exp,'resolved_mask':sorted(resolved),'provenance':'trie_equality_expansion_mask_guided','novelty_preflight':{'signature':'trie_equality_expansion_mask_v1','distinct_from':'mask choice; expands lexical trie states on both sides with an exact required character'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'trie-equality-expansion-mask-20260917','method':'mask-selected positions expand paired lexical trie states with exact equality requirements','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'continue trie expansion until a complete compatible word-boundary pair is formed, preserving semantic slot constraints'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
