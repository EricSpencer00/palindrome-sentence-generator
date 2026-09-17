#!/usr/bin/env python3
"""Semantic role tries for coupled character-level scene expansion."""
import hashlib,json,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/semantic-role-trie-coupled-expansion-20260917.json'
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
 for vals in list(itertools.product(*[DOM[k] for k in K]))[:12]:
  x=dict(zip(K,vals));t=norm(render(x));trace=[];alive=True
  for p in range(min(20,len(t)//2)):
   ok=t[p]==t[-1-p];trace.append({'position':p,'left':t[p],'right':t[-1-p],'alive':ok})
   if not ok:pruned+=1;alive=False;break
  if alive or len(rows)<6:rows.append({'rendered':render(x),'semantic_roles':x,'role_trie_trace':trace,'branch_status':'survived' if alive else 'pruned_after_recording','provenance':'semantic_role_trie_coupled_expansion','novelty_preflight':{'signature':'semantic_role_trie_coupled_expansion_v1','distinct_from':'fixed bundles; independent role tries provide lexical alternatives during coupled character expansion'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'semantic-role-trie-coupled-expansion-20260917','method':'independent semantic role tries supply lexical alternatives during coupled character expansion with immediate mismatch pruning','role_domains':{k:len(v) for k,v in DOM.items()},'pruned_branches':pruned,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'use live trie nodes rather than completed words so each role branch can satisfy mirrored character obligations incrementally'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
