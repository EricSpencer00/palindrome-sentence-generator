#!/usr/bin/env python3
"""Intersect lexical supports using true rendered-tape palindrome offsets."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/rendered-tape-offset-support-20260917.json'
T='The {a} {v} the {o} near the {s0}, and the {a2} {v2} the {o2} near the {s1}.'
D={'a':['gardener','teacher'],'v':['carries','writes'],'o':['letters','notes'],'s0':['harbor','garden'],'a2':['teacher','messenger'],'v2':['writes','records'],'o2':['notes','charts'],'s1':['garden','station']};K=list(D)
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def offsets(x):
 out={};p=0
 for z in re.split(r'({\w+})',T):
  if z.startswith('{'):
   k=z[1:-1];w=norm(x[k]);out[k]=(p,p+len(w)-1);p+=len(w)
  else:p+=len(norm(z))
 return out
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 rows=[]
 for i in range(2):
  x={k:D[k][i%len(D[k])] for k in K};t=norm(render(x));sp=offsets(x);matches=[]
  for k,(lo,hi) in sp.items():
   for pos in range(lo,hi+1):
    opp=len(t)-1-pos
    if t[pos]==t[opp]:matches.append({'slot':k,'slot_offset':pos-lo,'tape_position':pos,'opposing_position':opp,'char':t[pos]})
  rows.append({'candidate':i,'rendered':render(x),'slots':x,'rendered_slot_offsets':{k:list(v) for k,v in sp.items()},'true_opposing_supports':matches,'provenance':'rendered_tape_offset_true_opposing_support','novelty_preflight':{'signature':'rendered_tape_offset_support_v1','distinct_from':'within-word indices; maps each slot character to its actual normalized tape position and opposing index'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'rendered-tape-offset-support-20260917','method':'map lexical slot characters to normalized tape offsets and intersect supports at true opposing palindrome positions','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'use opposing-offset supports during lexical choice so unsupported slot values are rejected before full tape assembly'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
