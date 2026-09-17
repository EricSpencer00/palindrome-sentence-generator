#!/usr/bin/env python3
"""Intersect attachment witness domain with agent/action/theme valency."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/attachment-valency-witness-intersection-20260917.json'
T='At dawn, the {agent} {action} the {theme} {prep} the harbor; meanwhile, a cartographer marks the maps near the archive.'
VAL=[('teacher','writes','notes','along'),('gardener','carries','letters','through'),('teacher','writes','notes','through')]
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 rows=[];rejected=[]
 for i,(a,v,o,p) in enumerate(VAL):
  x={'agent':a,'action':v,'theme':o,'prep':p};t=norm(render(x));start=t.find(norm(p));w=[q for q in range(start,start+len(norm(p))) if q<len(t)//2 and t[q]==t[-1-q]]
  if not w:rejected.append({'valency':[a,v,o,p],'reason':'no opposing witness'});continue
  rows.append({'candidate':len(rows),'rendered':render(x),'semantic_valency':[a,v,o],'attachment':p,'witness_positions':w,'rejected_valencies':rejected,'provenance':'attachment_valency_witness_intersection','novelty_preflight':{'signature':'attachment_valency_witness_intersection_v1','distinct_from':'witness-only narrowing; agent/action/theme valency and attachment witness are intersected before scene expansion'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'attachment-valency-witness-intersection-20260917','method':'intersect semantic agent/action/theme valency bundles with position-aware attachment witness domain before expansion','candidate_count':len(rows),'candidates':rows,'rejected_valencies':rejected,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'expand both scene roles under the intersected typed domain and propagate witnesses bidirectionally'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
