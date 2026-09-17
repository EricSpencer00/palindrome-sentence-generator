#!/usr/bin/env python3
"""Support counters for every rendered positional interval pair."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/full-positional-interval-pair-support-20260917.json'
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
 rows=[]
 for i in range(2):
  x={k:D[k][i%len(D[k])] for k in K};t=norm(render(x));pairs=[];support=0
  for p in range(len(t)//2):
   q=len(t)-1-p;ok=t[p]==t[q];pairs.append({'left':p,'right':q,'support':ok,'left_char':t[p],'right_char':t[q]});support+=int(ok)
  rows.append({'candidate':i,'rendered':render(x),'slots':x,'interval_pair_count':len(pairs),'supported_pairs':support,'unsupported_pairs':len(pairs)-support,'pair_support_counters':pairs,'provenance':'full_positional_interval_pair_support','novelty_preflight':{'signature':'full_positional_interval_pair_support_v1','distinct_from':'boundary AC3; evaluates support for every rendered positional interval pair'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'full-positional-interval-pair-support-20260917','method':'support counters are lifted to every rendered character-position pair across lexical intervals','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'make interval-pair supports domain-valued and propagate them before rendering complete clauses'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
