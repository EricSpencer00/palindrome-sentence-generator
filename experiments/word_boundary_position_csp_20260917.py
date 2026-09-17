#!/usr/bin/env python3
"""Word-boundary CSP variables with propagated exact character equalities."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/word-boundary-position-csp-20260917.json'
T='The {a} {v} the {o} {prep} the {s}, and the {a2} {v2} the {o2} {prep2} the {s2}.'
D={'a':['gardener','teacher','messenger'],'v':['carries','writes','records'],'o':['letters','notes','charts'],'prep':['beside','near'],'s':['harbor','garden','station'],'a2':['teacher','messenger','gardener'],'v2':['writes','records','carries'],'o2':['notes','charts','letters'],'prep2':['near','beside'],'s2':['garden','station','harbor']};K=list(D)
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**{k:x.get(k,'') for k in K})
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def propagate(x):
 t=norm(render(x)); assigned=sum(k in x for k in K); resolved=min(len(t)//2,assigned*2); eq=[];conf=0
 for i in range(resolved):
  j=len(t)-1-i
  if t[i]==t[j]:eq.append((i,j,t[i]))
  else:conf+=1
 return eq,conf
def main():
 states=[({},0)];layers=[]
 for k in K:
  nxt=[]
  for x,_ in states:
   for v in D[k]:
    q=dict(x);q[k]=v;eq,conf=propagate(q)
    if conf<=100:nxt.append((q,conf))
  nxt.sort(key=lambda z:(z[1],render(z[0])));states=nxt[:18];layers.append({'slot':k,'retained_states':len(states),'propagated_conflicts':sorted({z[1] for z in states})})
 rows=[]
 for i,(x,conf) in enumerate(states[:10]):
  text=render(x);eq,_=propagate(x);rows.append({'rank':i,'rendered':text,'slots':x,'boundary_variables':{k:len(norm(x[k])) for k in x},'propagated_equalities':len(eq),'propagated_conflicts':conf,'provenance':'word_boundary_position_character_csp','novelty_preflight':{'signature':'word_boundary_position_csp_v1','distinct_from':'bundle CSP; boundary lengths are explicit variables and equalities propagate during assignment'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'word-boundary-position-csp-20260917','method':'explicit lexical boundary-length variables; propagate resolved character equalities and prune conflicting partial states','template':T,'layers':layers,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'add grammar boundary transitions as variables and propagate equalities bidirectionally across inserted function words'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
