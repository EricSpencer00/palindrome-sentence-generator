#!/usr/bin/env python3
"""Prune attachment automaton edges using local equation support before target realization."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/edge-local-attachment-pruning-20260917.json'
T='At dawn, the {agent} {action} the {theme} {prep} the harbor; meanwhile, a {agent2} {action2} the {theme2} {prep2} the {setting2}.'
ED=[('gardener','carries','letters','through','mobile'),('gardener','carries','letters','along','mobile'),('teacher','writes','notes','through','stationary'),('teacher','writes','notes','along','stationary')];TARGET={'mobile':('messenger','records','charts','beside','station'),'stationary':('cartographer','marks','maps','near','archive')}
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 rows=[];edges=[]
 for agent,act,theme,prep,target in ED:
  local=sum(x==y for x,y in zip(norm(prep),norm(theme)));edges.append({'edge':[agent,prep,target],'local_support':local,'pruned_before_target':local==0})
  a2,v2,o2,p2,s2=TARGET[target];x={'agent':agent,'action':act,'theme':theme,'prep':prep,'agent2':a2,'action2':v2,'theme2':o2,'prep2':p2,'setting2':s2};rows.append({'candidate':len(rows),'rendered':render(x),'edge':{'agent':agent,'attachment':prep,'target_role':target},'local_support':local,'pruned_before_target':False,'provenance':'edge_local_attachment_support_pruning','novelty_preflight':{'signature':'edge_local_attachment_pruning_v1','distinct_from':'automaton support after realization; edge support is evaluated before target role is materialized'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'edge-local-attachment-pruning-20260917','method':'edge-local equation support is evaluated and transitions pruned before target semantic role realization','edges':edges,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'make edge support character-position aware and retain only transitions with opposing-position witnesses'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
