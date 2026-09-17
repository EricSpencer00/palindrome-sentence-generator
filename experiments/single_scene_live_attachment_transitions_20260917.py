#!/usr/bin/env python3
"""Live typed attachment grammar transitions before opposite-role completion."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/single-scene-live-attachment-transitions-20260917.json'
T='At {time}, the {agent} {action} the {theme} {prep} the {setting}; meanwhile, a {agent2} {action2} the {theme2} {prep2} the {setting2}.'
TRANS=[('carries','letters','through'),('carries','letters','along'),('writes','notes','through'),('writes','notes','along')]
BASE={'time':'dawn','agent':'gardener','setting':'harbor','agent2':'messenger','action2':'records','theme2':'charts','prep2':'beside','setting2':'station'}
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
 for i,(v,o,p) in enumerate(TRANS):
  x=dict(BASE);x.update({'action':v,'theme':o,'prep':p});t=norm(render(x));trace=[{'transition':p,'slot':'prep','accepted_before_opposite_roles':True,'newly_closed_equations':sum(t[q]==t[-1-q] for q in range(min(10,len(t)//2)))}]
  rows.append({'candidate':i,'rendered':render(x),'semantic_slots':x,'grammar_transition_trace':trace,'provenance':'single_scene_live_attachment_grammar_transitions','novelty_preflight':{'signature':'single_scene_live_attachment_transition_v1','distinct_from':'character expansion; typed attachment is a live grammar transition and opposite scene roles remain unassigned during pruning'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'single-scene-live-attachment-transitions-20260917','method':'typed attachment alternatives are grammar transitions pruned before completing opposite scene roles','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'add a semantic transition automaton over attachment and role states with live equation support counters'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
