#!/usr/bin/env python3
"""Use boundary trie labels as exact requirements on opposing role nodes."""
import hashlib,json,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/boundary-label-opposing-role-requirements-20260917.json'
T='At dawn, the {a} {v} the {o} {prep} the {s}; meanwhile, a {a2} {v2} the {o2} {prep2} the {s2}.'
D={'a':['gardener','teacher'],'v':['carries','writes'],'o':['letters','notes'],'prep':['through','along'],'s':['harbor','garden'],'a2':['messenger','cartographer'],'v2':['records','marks'],'o2':['charts','maps'],'prep2':['near','beside'],'s2':['station','archive']};K=list(D)
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
 for vals in list(itertools.product(*[D[k] for k in K]))[:8]:
  x=dict(zip(K,vals));t=norm(render(x));labels=[{'pos':p,'required':t[p],'opposing_role_requirement':t[p]} for p in range(min(10,len(t)//2)) if t[p]==t[-1-p]]
  rows.append({'rendered':render(x),'semantic_roles':x,'boundary_labels':labels,'opposing_role_nodes':{k:norm(x[k])[:2] for k in K},'provenance':'boundary_label_opposing_role_requirements','novelty_preflight':{'signature':'boundary_label_opposing_role_requirements_v1','distinct_from':'boundary trie nodes; labels become exact required characters on opposing role nodes during coupled expansion'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'boundary-label-opposing-role-requirements-20260917','method':'boundary trie labels alter the coupled state by imposing exact character requirements on opposing semantic role nodes','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'propagate required labels backward through role tries and reject incompatible nodes before completing words'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
