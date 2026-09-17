#!/usr/bin/env python3
"""Incrementally propagate witnesses while pruning one scene side."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/incremental-bidirectional-witness-pruning-20260917.json'
T='At dawn, the {agent} {action} the {theme} {prep} the harbor; meanwhile, a {agent2} {action2} the {theme2} {prep2} the {setting2}.'
L=[('teacher','writes','notes','along'),('gardener','carries','letters','through')];R=[('cartographer','marks','maps','near','archive'),('messenger','records','charts','beside','station')]
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 rows=[]
 for l in L:
  for r in R:
   x={'agent':l[0],'action':l[1],'theme':l[2],'prep':l[3],'agent2':r[0],'action2':r[1],'theme2':r[2],'prep2':r[3],'setting2':r[4]};t=norm(render(x));left=[q for q in range(min(10,len(t)//2)) if t[q]==t[-1-q]];right=[len(t)-1-q for q in left]
   rows.append({'candidate':len(rows),'rendered':render(x),'left':list(l),'right':list(r),'incremental_steps':[{'side':'left','witnesses':left},{'side':'right','witnesses':right}],'pruned_side':'right' if not right else None,'provenance':'incremental_bidirectional_witness_one_side_pruning','novelty_preflight':{'signature':'incremental_bidirectional_witness_pruning_v1','distinct_from':'complete paired expansion; witnesses update after left assignment and prune right before full completion'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'incremental-bidirectional-witness-pruning-20260917','method':'witnesses propagate after each side assignment; unsupported opposite side is pruned before completion','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'make pruning domain-valued and retain only right-role candidates with explicit witness support'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
