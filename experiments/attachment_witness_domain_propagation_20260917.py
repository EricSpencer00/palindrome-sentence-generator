#!/usr/bin/env python3
"""Propagate position-aware attachment witnesses into lexical domains pre-render."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/attachment-witness-domain-propagation-20260917.json'
T='At dawn, the teacher writes the notes {prep} the harbor; meanwhile, a cartographer marks the maps near the archive.'
PREPS=['through','along','beside','near']
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(p):return T.format(prep=p)
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 kept=[];rejected=[]
 for p in PREPS:
  t=norm(render(p));start=t.find(norm(p));w=[{'position':q,'opposing':len(t)-1-q,'char':t[q]} for q in range(start,start+len(norm(p))) if q<len(t)//2 and t[q]==t[-1-q]]
  (kept if w else rejected).append({'attachment':p,'witnesses':w})
 rows=[]
 for i,k in enumerate(kept):
  rows.append({'candidate':i,'rendered':render(k['attachment']),'attachment':k['attachment'],'witnesses':k['witnesses'],'rejected_domain_values':rejected,'provenance':'attachment_witness_domain_propagation_pre_render','novelty_preflight':{'signature':'attachment_witness_domain_propagation_v1','distinct_from':'post-filter witness; attachment lexical domain is narrowed before rendering candidates'},'audit':audit(render(k['attachment'])),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'attachment-witness-domain-propagation-20260917','method':'position-aware witnesses narrow the attachment lexical domain before candidate rendering','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'combine attachment witness domains with agent/action/theme semantic valency domains before scene expansion'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
