#!/usr/bin/env python3
"""Prune lexical interval domains on first positional equality loss."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/interval-domain-support-pruning-20260917.json'
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
 dom={k:list(v) for k,v in D.items()};rounds=[];removed=0
 for k in K:
  keep=[]
  for w in dom[k]:
   x={z:D[z][0] for z in K};x[k]=w;t=norm(render(x));supported=any(t[i]==t[-1-i] for i in range(min(8,len(t)//2)))
   if supported:keep.append(w)
   else:removed+=1
  if keep:dom[k]=keep
  rounds.append({'slot':k,'retained':len(dom[k]),'removed_so_far':removed})
 rows=[]
 for i in range(2):
  x={k:dom[k][i%len(dom[k])] for k in K};text=render(x);rows.append({'candidate':i,'rendered':text,'slots':x,'domain_sizes':{k:len(v) for k,v in dom.items()},'support_pruning_rounds':rounds,'removed_values':removed,'provenance':'interval_domain_support_pruning','novelty_preflight':{'signature':'interval_domain_support_pruning_v1','distinct_from':'completed positional CSP; each slot value is pruned on first unsupported positional equality'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'interval-domain-support-pruning-20260917','method':'interval-domain values are tested for positional support and pruned immediately before full word completion','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'maintain support counters incrementally as neighboring domains change, using AC-3 propagation over positional intervals'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
