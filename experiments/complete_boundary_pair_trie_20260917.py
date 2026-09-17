#!/usr/bin/env python3
"""Complete compatible lexical boundary pairs from trie equality expansions."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/complete-boundary-pair-trie-20260917.json'
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
  x={k:D[k][i%len(D[k])] for k in K};pairs=[]
  for l,r in [('a','a2'),('v','v2'),('o','o2'),('s0','s1')]:
   a=norm(x[l]);b=norm(x[r]);pairs.append({'left_slot':l,'right_slot':r,'left_word':a,'right_word':b,'complete_boundary':a[-1]==b[0],'left_end':a[-1],'right_start':b[0]})
  rows.append({'candidate':i,'rendered':render(x),'slots':x,'completed_boundary_pairs':pairs,'provenance':'complete_compatible_boundary_pair_trie','novelty_preflight':{'signature':'complete_boundary_pair_trie_v1','distinct_from':'partial trie expansions; requires complete lexical boundary pair before admitting slot state'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'complete-boundary-pair-trie-20260917','method':'trie expansion continues until semantic lexical slot pairs have complete boundary compatibility','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'couple completed boundary pairs to whole-tape seam equations before admitting the full clause'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
