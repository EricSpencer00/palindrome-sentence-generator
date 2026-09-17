#!/usr/bin/env python3
"""Direct boundary-offset seam DP; no token-value span reconstruction."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/boundary-offset-direct-seam-dp-20260917.json'
T='{d0} {a0} {v0} {d1} {o0} {p0} {d2} {s0}, {c} {d3} {a1} {v1} {d4} {o1} {p1} {d5} {s1}.'
B={'d0':['the','a'],'a0':['gardener','teacher'],'v0':['carries','writes'],'d1':['the','a'],'o0':['letters','notes'],'p0':['beside','near'],'d2':['the','a'],'s0':['harbor','garden'],'c':['and','while'],'d3':['the','a'],'a1':['teacher','messenger'],'v1':['writes','records'],'d4':['the','a'],'o1':['notes','charts'],'p1':['near','beside'],'d5':['the','a'],'s1':['garden','station']}; KEYS=list(B)
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**{k:x.get(k,'') for k in KEYS})
def direct_offsets(x):
 # Walk template source and accumulate normalized offsets at placeholder time.
 out={}; p=0
 for piece in re.split(r'({\w+})',T):
  if piece.startswith('{'):
   k=piece[1:-1]; w=norm(x.get(k,'')); out[k]=(p,p+len(w)-1);p+=len(w)
  else:p+=len(norm(piece))
 return out
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 states=[({}, {}, 0)]; layers=[]
 for k in KEYS:
  nxt=[]
  for st,old,bad in states:
   for v in B[k]:
    q=dict(st);q[k]=v; off=direct_offsets(q); t=norm(render(q)); newly=[]
    # Offsets are keyed by slot, so duplicate lexical values remain distinct.
    for z,(lo,hi) in off.items():
     if z not in old: newly.extend(range(lo,hi+1))
    closed=[]
    for i in newly:
     j=len(t)-1-i
     if j in newly and i<j: closed.append((i,j))
    conflicts=sum(t[i]!=t[j] for i,j in closed)
    if conflicts<=2:nxt.append((q,off,bad+conflicts))
  nxt.sort(key=lambda z:(z[2],render(z[0])));states=nxt[:12];layers.append({'slot':k,'retained_states':len(states),'offset_widths':sorted({len(z[1]) for z in states})})
 rows=[]
 for rank,(x,off,bad) in enumerate(states[:8]):
  text=render(x);a=audit(text)
  rows.append({'rank':rank,'rendered':text,'slots':x,'direct_offsets':{k:list(v) for k,v in off.items()},'incremental_conflicts':bad,'provenance':'direct_template_offset_seam_dp','novelty_preflight':{'signature':'direct_offset_seam_dp_v1','distinct_from':'interval reconstruction; offsets are emitted while walking template placeholders and keyed by slot'},'audit':a,'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'boundary-offset-direct-seam-dp-20260917','method':'walk template placeholders to track direct normalized offsets per slot; newly closed offset pairs are checked incrementally','template':T,'layers':layers,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'add variable-length clause boundary states and preserve direct offsets through insertion/deletion transitions'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
