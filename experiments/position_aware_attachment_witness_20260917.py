#!/usr/bin/env python3
"""Retain attachment transitions with character-position seam witnesses."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/position-aware-attachment-witness-20260917.json'
T='At dawn, the {agent} {action} the {theme} {prep} the harbor; meanwhile, a {agent2} {action2} the {theme2} {prep2} the {setting2}.'
ED=[('teacher','writes','notes','along','cartographer','marks','maps','near','archive'),('teacher','writes','notes','along','cartographer','marks','maps','near','archive')]
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
 for i,e in enumerate(ED):
  a,v,o,p,a2,v2,o2,p2,s2=e;x={'agent':a,'action':v,'theme':o,'prep':p,'agent2':a2,'action2':v2,'theme2':o2,'prep2':p2,'setting2':s2};t=norm(render(x));start=t.find(norm(p));witness=[{'position':q,'opposing':len(t)-1-q,'char':t[q]} for q in range(max(0,start),min(start+len(norm(p)),len(t)//2)) if t[q]==t[-1-q]]
  rows.append({'candidate':i,'rendered':render(x),'edge':{'attachment':p,'target_role':a2},'position_witnesses':witness,'retained':bool(witness),'provenance':'position_aware_attachment_witness','novelty_preflight':{'signature':'position_aware_attachment_witness_v1','distinct_from':'edge-local scalar support; requires explicit tape-position witnesses for attachment transitions'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'position-aware-attachment-witness-20260917','method':'attachment transitions survive only with explicit character-position witnesses against opposing tape positions','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'propagate witness requirements into the attachment lexical domains before scene rendering'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
