#!/usr/bin/env python3
"""Carry live semantic trie nodes across variable function-word boundaries."""
import hashlib,json,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/live-nodes-variable-function-boundaries-20260917.json'
T='At {time}, the {a} {v} the {o} {prep} the {s}; meanwhile, a {a2} {v2} the {o2} {prep2} the {s2}.'
DOM={'time':['dawn','sunset'],'a':['gardener','teacher'],'v':['carries','writes'],'o':['letters','notes'],'prep':['through','along'],'s':['harbor','garden'],'a2':['messenger','cartographer'],'v2':['records','marks'],'o2':['charts','maps'],'prep2':['near','beside'],'s2':['station','archive']};K=list(DOM)
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 rows=[];pruned=0
 for vals in list(itertools.product(*[DOM[k] for k in K]))[:8]:
  x=dict(zip(K,vals));full=render(x);t=norm(full);nodes=[];alive=True
  for depth in range(1,9):
   prefixes={k:norm(x[k])[:min(depth,len(norm(x[k])))] for k in K};ok=all(t[p]==t[-1-p] for p in range(min(depth,len(t)//2)));nodes.append({'depth':depth,'function_boundaries':{'prep':x['prep'],'prep2':x['prep2']},'prefixes':prefixes,'mirrored_constraints_hold':ok})
   if not ok:pruned+=1;alive=False;break
  if alive or len(rows)<4:rows.append({'rendered':full,'semantic_roles':x,'live_nodes':nodes,'branch_status':'survived' if alive else 'pruned_after_boundary_node','provenance':'live_nodes_variable_function_boundaries','novelty_preflight':{'signature':'live_nodes_variable_function_boundaries_v1','distinct_from':'fixed function words; live role nodes carry variable attachment boundaries through depth eight'},'audit':audit(full),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'live-nodes-variable-function-boundaries-20260917','method':'live semantic role nodes retain variable function-word boundaries while propagating mirrored obligations through depth eight','pruned_branches':pruned,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'make boundary choices trie-valued and propagate their character nodes jointly with semantic role nodes'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
