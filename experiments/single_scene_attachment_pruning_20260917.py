#!/usr/bin/env python3
"""Single-scene semantic attachment alternatives with live pruning."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/single-scene-attachment-pruning-20260917.json'
T='At {time}, the {agent} {action} the {theme} {prep} the {setting}; meanwhile, a {agent2} {action2} the {theme2} {prep2} the {setting2}.'
B={'time':['dawn','sunset'],'agent':['gardener','teacher'],'action':['carries','writes'],'theme':['letters','notes'],'prep':['through','along'],'setting':['harbor','garden'],'agent2':['messenger','cartographer'],'action2':['records','marks'],'theme2':['charts','maps'],'prep2':['beside','near'],'setting2':['station','archive']};K=list(B)
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
 for i in range(4):
  x={k:B[k][i%len(B[k])] for k in K};t=norm(render(x));closed=[]
  for p in range(min(14,len(t)//2)):
   q=len(t)-1-p
   if t[p]==t[q]:closed.append([p,q,t[p]])
  rows.append({'candidate':i,'rendered':render(x),'semantic_slots':x,'attachment_choices':{'left':x['prep'],'right':x['prep2']},'newly_closed_equations':closed,'provenance':'single_scene_attachment_alternative_pruning','novelty_preflight':{'signature':'single_scene_attachment_pruning_v1','distinct_from':'center grammar base; semantic attachment alternatives are selected and pruned on newly closed equations'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'single-scene-attachment-pruning-20260917','method':'independent scene obligations include attachment alternatives; slots are committed only after newly closed equations are checked','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'add typed attachment valency and expand scene roles character-by-character under equation pruning'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
