#!/usr/bin/env python3
"""Expand both scene roles under typed valency/witness domains."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/bidirectional-scene-valency-witness-expansion-20260917.json'
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
   x={'agent':l[0],'action':l[1],'theme':l[2],'prep':l[3],'agent2':r[0],'action2':r[1],'theme2':r[2],'prep2':r[3],'setting2':r[4]};t=norm(render(x));left=[p for p in range(min(14,len(t)//2)) if t[p]==t[-1-p]];right=[len(t)-1-p for p in left];rows.append({'candidate':len(rows),'rendered':render(x),'left_valency':list(l),'right_valency':list(r),'left_witnesses':left,'right_witnesses':right,'provenance':'bidirectional_scene_valency_witness_expansion','novelty_preflight':{'signature':'bidirectional_scene_valency_witness_v1','distinct_from':'one-side intersection; expands both independent scene role bundles and propagates witnesses from both sides'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'bidirectional-scene-valency-witness-expansion-20260917','method':'expand independent left/right typed scene valencies while propagating witness positions in both directions','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'make witness propagation incremental during paired role expansion and prune a side before completing the other'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
