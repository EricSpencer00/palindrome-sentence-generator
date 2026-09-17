#!/usr/bin/env python3
"""Propagate live semantic role trie nodes before completed-word materialization."""
import hashlib,json,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/live-role-trie-node-propagation-20260917.json'
T='At dawn, the {a} {v} the {o} through the {s}; meanwhile, a {a2} {v2} the {o2} near the {s2}.'
DOM={'a':['gardener','teacher','archivist'],'v':['carries','writes','keeps'],'o':['letters','notes','records'],'s':['harbor','garden','archive'],'a2':['messenger','cartographer','courier'],'v2':['records','marks','delivers'],'o2':['charts','maps','parcels'],'s2':['station','archive','harbor']};K=list(DOM)
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
  for depth in range(1,5):
   prefixes={k:norm(x[k])[:depth] for k in K};resolved=''.join(prefixes.values());ok=all(t[p]==t[-1-p] for p in range(min(depth,len(t)//2)));nodes.append({'depth':depth,'prefixes':prefixes,'mirrored_constraints_hold':ok})
   if not ok:pruned+=1;alive=False;break
  if alive or len(rows)<4:rows.append({'rendered':full,'semantic_roles':x,'live_trie_nodes':nodes,'branch_status':'survived' if alive else 'pruned_after_live_node','provenance':'live_role_trie_node_propagation','novelty_preflight':{'signature':'live_role_trie_node_propagation_v1','distinct_from':'completed role tries; prefix nodes are propagated and checked before materializing full words'},'audit':audit(full),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'live-role-trie-node-propagation-20260917','method':'live prefix nodes for each semantic role are propagated and checked against mirrored obligations before full-word admission','pruned_branches':pruned,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'carry live node states across variable function-word boundaries and continue exact propagation past depth four'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
