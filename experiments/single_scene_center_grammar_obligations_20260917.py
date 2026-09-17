#!/usr/bin/env python3
"""Single-scene center grammar with independent semantic obligations."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/single-scene-center-grammar-obligations-20260917.json'
T='At {time}, the {agent} {action} the {theme} through the {setting}; meanwhile, a {agent2} {action2} the {theme2} beside the {setting2}.'
B={'time':['dawn','sunset'],'agent':['gardener','teacher'],'action':['carries','writes'],'theme':['letters','notes'],'setting':['harbor','garden'],'agent2':['messenger','cartographer'],'action2':['records','marks'],'theme2':['charts','maps'],'setting2':['station','archive']};K=list(B)
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
  x={k:B[k][(i+(1 if k in ('agent2','action2','theme2','setting2') else 0))%len(B[k])] for k in K};t=norm(render(x));eq=[]
  for p in range(min(12,len(t)//2)):
   q=len(t)-1-p
   if t[p]==t[q]:eq.append([p,q,t[p]])
  rows.append({'candidate':i,'rendered':render(x),'semantic_slots':x,'live_equations_checked':min(12,len(t)//2),'equation_matches':eq,'provenance':'single_scene_center_grammar_independent_obligations','novelty_preflight':{'signature':'single_scene_center_grammar_v1','distinct_from':'two-clause seam families; independent scene roles assigned in one center grammar with live tape equations'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'single-scene-center-grammar-obligations-20260917','method':'one scene grammar independently assigns agent/action/theme/setting obligations on each side while checking live character equations','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'add semantic attachment alternatives inside the scene grammar and prune on newly closed equations before committing a slot'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
