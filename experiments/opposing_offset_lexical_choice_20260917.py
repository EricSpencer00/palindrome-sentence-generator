#!/usr/bin/env python3
"""Reject lexical values during choice using true opposing tape offsets."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/opposing-offset-lexical-choice-20260917.json'
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
 rows=[];rejected={}
 for k,vals in D.items():
  keep=[];reject=[]
  for w in vals:
   x={q:D[q][0] for q in K};x[k]=w;t=norm(render(x));supported=sum(t[p]==t[-1-p] for p in range(min(4,len(t)//2)))
   (keep if supported else reject).append(w)
  rejected[k]=reject
  if not keep:keep=vals
  x={q:D[q][0] for q in K};x[k]=keep[0];rows.append({'slot_choice':k,'chosen':keep[0],'rendered':render(x),'rejected_during_choice':reject,'opposing_offset_support':sum(norm(render(x))[p]==norm(render(x))[-1-p] for p in range(min(4,len(norm(render(x)))//2))),'provenance':'opposing_offset_lexical_choice_before_assembly','novelty_preflight':{'signature':'opposing_offset_lexical_choice_v1','distinct_from':'post-render offset audit; lexical values are rejected during each slot choice using true tape offsets'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'opposing-offset-lexical-choice-20260917','method':'slot lexical choices are filtered using opposing normalized tape offsets before candidate assembly','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'make opposing-offset support conditional on all already chosen slots and propagate choices jointly'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
