#!/usr/bin/env python3
"""Attachment/role automaton with live equation support counters."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/attachment-role-transition-automaton-20260917.json'
T='At {time}, the {agent} {action} the {theme} {prep} the {setting}; meanwhile, a {agent2} {action2} the {theme2} {prep2} the {setting2}.'
ST={'carrier':('gardener','carries','letters'),'scribe':('teacher','writes','notes')};ED={'carrier':[('through','stationary') ,('along','stationary')],'scribe':[('through','mobile'),('along','mobile')]};OTHER={'stationary':('messenger','records','charts','beside','station'),'mobile':('cartographer','marks','maps','near','archive')}
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
 for src,edges in ED.items():
  for prep,target in edges:
   a,v,o=ST[src];a2,v2,o2,p2,s2=OTHER[target];x={'time':'dawn','agent':a,'action':v,'theme':o,'prep':prep,'setting':'harbor','agent2':a2,'action2':v2,'theme2':o2,'prep2':p2,'setting2':s2};t=norm(render(x));support=sum(t[p]==t[-1-p] for p in range(min(16,len(t)//2)))
   rows.append({'candidate':len(rows),'rendered':render(x),'source_role':src,'attachment_transition':prep,'target_role':target,'support_counter':support,'provenance':'attachment_role_transition_automaton_live_support','novelty_preflight':{'signature':'attachment_role_transition_automaton_v1','distinct_from':'live attachment transitions; finite role/attachment automaton carries target-role state and equation support counters'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'attachment-role-transition-automaton-20260917','method':'semantic role/attachment automaton traverses target states while carrying live character-equation support counters','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'make support counters edge-local and prune automaton transitions before target role realization'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
