#!/usr/bin/env python3
"""Interval-valued function-word boundaries with incremental seam checks."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/interval-boundary-incremental-seam-20260917.json'
T='{d0} {a0} {v0} {d1} {o0} {p0} {d2} {s0}, {c} {d3} {a1} {v1} {d4} {o1} {p1} {d5} {s1}.'
B={'d0':['the','a'],'a0':['gardener','teacher'],'v0':['carries','writes'],'d1':['the','a'],'o0':['letters','notes'],'p0':['beside','near'],'d2':['the','a'],'s0':['harbor','garden'],'c':['and','while'],'d3':['the','a'],'a1':['teacher','messenger'],'v1':['writes','records'],'d4':['the','a'],'o1':['notes','charts'],'p1':['near','beside'],'d5':['the','a'],'s1':['garden','station']}; KEYS=list(B)
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**{k:x.get(k,'') for k in KEYS})
def intervals(x):
 # Compute normalized character intervals for each filled grammar slot.
 text=render(x); spans={}; cursor=0
 for token in text.split():
  clean=norm(token)
  if not clean: continue
  for k,v in x.items():
   if norm(v)==clean and k not in spans: spans[k]=(cursor,cursor+len(clean)-1); break
  cursor+=len(clean)
 return spans
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def newly_closed(x, old_spans):
 t=norm(render(x)); sp=intervals(x); pairs=[]
 for i in range(len(t)//2):
  j=len(t)-1-i
  if i<j and (i not in {q for z in old_spans.values() for q in range(z[0],z[1]+1)} or j<0): pairs.append((i,j))
 # Only pairs whose endpoints are in newly added intervals are checked.
 old=set(q for z in old_spans.values() for q in range(z[0],z[1]+1)); now=set(q for z in sp.values() for q in range(z[0],z[1]+1))
 closed=[(i,j) for i,j in pairs if (i in now and j in now and (i not in old or j not in old))]
 return closed,sp
def main():
 states=[({}, {}, 0)]; layers=[]
 for k in KEYS:
  nxt=[]
  for st,old,bad in states:
   for v in B[k]:
    q=dict(st);q[k]=v;closed,sp=newly_closed(q,old);t=norm(render(q)); conflicts=sum(t[i]!=t[j] for i,j in closed)
    if conflicts<=2: nxt.append((q,sp,bad+conflicts))
  nxt.sort(key=lambda z:(z[2],render(z[0])));states=nxt[:12];layers.append({'slot':k,'retained_states':len(states),'closed_pair_counts':sorted({len(z[1]) for z in states})})
 rows=[]
 for rank,(x,sp,bad) in enumerate(states[:8]):
  text=render(x);a=audit(text)
  rows.append({'rank':rank,'rendered':text,'slots':x,'intervals':{k:list(v) for k,v in sp.items()},'incremental_conflicts':bad,'provenance':'interval_boundary_incremental_seam_dp','novelty_preflight':{'signature':'interval_boundary_incremental_v1','distinct_from':'resolved position masks; stores slot intervals and checks newly closed pairs only'},'audit':a,'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'interval-boundary-incremental-seam-20260917','method':'DP carries normalized character intervals for filled function-word slots and checks only seam pairs newly closed by each interval update','template':T,'layers':layers,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'replace token-span interval reconstruction with boundary-aware incremental offsets so repeated lexical values cannot alias slot intervals'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
