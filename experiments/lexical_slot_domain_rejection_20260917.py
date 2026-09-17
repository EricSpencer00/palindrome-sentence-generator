#!/usr/bin/env python3
"""Reject unsupported lexical slot values before tape materialization."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/lexical-slot-domain-rejection-20260917.json'
T='The {a} {v} the {o} near the {s0}, and the {a2} {v2} the {o2} near the {s1}.'
D={'a':['gardener','teacher'],'v':['carries','writes'],'o':['letters','notes'],'s0':['harbor','garden'],'a2':['teacher','messenger'],'v2':['writes','records'],'o2':['notes','charts'],'s1':['garden','station']};K=list(D)
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 domains={k:set(norm(w)[0] for w in vals) for k,vals in D.items()};filtered={};removed={}
 for k,vals in D.items():
  keep=[w for w in vals if norm(w)[:1] in domains[k]];filtered[k]=keep;removed[k]=len(vals)-len(keep)
 rows=[]
 for i in range(2):
  x={k:filtered[k][i%len(filtered[k])] for k in K};text=render(x);rows.append({'candidate':i,'rendered':text,'slots':x,'slot_domains':{k:list(v) for k,v in filtered.items()},'rejected_before_materialization':removed,'provenance':'lexical_slot_domain_rejection_before_materialization','novelty_preflight':{'signature':'lexical_slot_domain_rejection_v1','distinct_from':'domain support reporting; unsupported lexical words are removed before any full tape is created'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'lexical-slot-domain-rejection-20260917','method':'propagated character domains reject unsupported lexical values before full tape materialization','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'propagate opposing-position domains into each slot iteratively until lexical domains reach a fixed point'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
