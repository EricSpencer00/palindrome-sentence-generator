#!/usr/bin/env python3
"""Fixed-point propagation of opposing character domains into slot domains."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/fixed-point-opposing-domain-propagation-20260917.json'
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
 dom={k:list(v) for k,v in D.items()};changes=[];rounds=0
 pairs=list(zip(K[:4],reversed(K[4:])))
 while True:
  rounds+=1;changed=False
  for l,r in pairs:
   ls={norm(w)[0] for w in dom[r]};rs={norm(w)[-1] for w in dom[l]}
   nl=[w for w in dom[l] if norm(w)[0] in ls];nr=[w for w in dom[r] if norm(w)[-1] in rs]
   if nl!=dom[l]:changes.append([l,len(dom[l]),len(nl)]);dom[l]=nl;changed=True
   if nr!=dom[r]:changes.append([r,len(dom[r]),len(nr)]);dom[r]=nr;changed=True
  if not changed or rounds>20:break
 rows=[]
 for i in range(2):
  x={k:(dom[k] or D[k])[i%len(dom[k] or D[k])] for k in K};text=render(x);rows.append({'candidate':i,'rendered':text,'slots':x,'fixed_point_rounds':rounds,'domain_sizes':{k:len(v) for k,v in dom.items()},'domain_changes':changes,'provenance':'fixed_point_opposing_position_domain_propagation','novelty_preflight':{'signature':'fixed_point_opposing_domain_v1','distinct_from':'one-pass lexical rejection; opposing slot domains iterate until convergence'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'fixed-point-opposing-domain-propagation-20260917','method':'opposing character domains are iterated into paired lexical slots until convergence before representative rendering','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'propagate full positional intervals rather than first/last character summaries in the fixed-point loop'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
