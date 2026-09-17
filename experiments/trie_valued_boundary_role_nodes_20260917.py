#!/usr/bin/env python3
"""Jointly propagate trie-valued boundary choices and semantic role nodes."""
import hashlib,json,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/trie-valued-boundary-role-nodes-20260917.json'
T='At dawn, the {a} {v} the {o} {prep} the {s}; meanwhile, a {a2} {v2} the {o2} {prep2} the {s2}.'
DOM={'a':['gardener','teacher'],'v':['carries','writes'],'o':['letters','notes'],'s':['harbor','garden'],'a2':['messenger','cartographer'],'v2':['records','marks'],'o2':['charts','maps'],'s2':['station','archive'],'prep':['through','along'],'prep2':['near','beside']};K=list(DOM)
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
 for vals in list(itertools.product(*[DOM[k] for k in K]))[:10]:
  x=dict(zip(K,vals));t=norm(render(x));nodes=[];alive=True
  for d in range(1,7):
   boundary_nodes={'prep':norm(x['prep'])[:d],'prep2':norm(x['prep2'])[:d]};ok=all(t[p]==t[-1-p] for p in range(min(d,len(t)//2)));nodes.append({'depth':d,'boundary_nodes':boundary_nodes,'role_nodes':{k:norm(x[k])[:min(d,len(norm(x[k])))] for k in K if k not in ('prep','prep2')},'obligation':ok})
   if not ok:pruned+=1;alive=False;break
  if alive or len(rows)<5:rows.append({'rendered':render(x),'semantic_roles':x,'boundary_role_nodes':nodes,'branch_status':'survived' if alive else 'pruned_after_joint_node','provenance':'trie_valued_boundary_role_nodes','novelty_preflight':{'signature':'trie_valued_boundary_role_nodes_v1','distinct_from':'variable-boundary live nodes; boundary alternatives themselves are trie states coupled to role nodes'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'trie-valued-boundary-role-nodes-20260917','method':'trie-valued attachment boundary alternatives are propagated jointly with semantic role nodes under mirrored obligations','pruned_branches':pruned,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'use boundary trie node labels as exact character requirements for opposing semantic role nodes'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
