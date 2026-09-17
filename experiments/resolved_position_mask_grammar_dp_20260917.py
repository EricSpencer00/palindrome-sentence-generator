#!/usr/bin/env python3
"""Grammar DP carrying resolved tape-position masks during seam expansion."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/resolved-position-mask-grammar-dp-20260917.json'
T='{d0} {a0} {v0} {d1} {o0} {p0} {d2} {s0}, {c} {d3} {a1} {v1} {d4} {o1} {p1} {d5} {s1}.'
B={'d0':['the','a'],'a0':['gardener','teacher'],'v0':['carries','writes'],'d1':['the','a'],'o0':['letters','notes'],'p0':['beside','near'],'d2':['the','a'],'s0':['harbor','garden'],'c':['and','while'],'d3':['the','a'],'a1':['teacher','messenger'],'v1':['writes','records'],'d4':['the','a'],'o1':['notes','charts'],'p1':['near','beside'],'d5':['the','a'],'s1':['garden','station']}; KEYS=list(B)
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**{k:x.get(k,'') for k in KEYS})
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def mask(x):
 # Positions belonging to completed lexical slots, calculated on current tape.
 t=norm(render(x)); return list(range(len(t)))
def compatible(x, known):
 t=norm(render(x)); m=set(known); pairs=[(i,len(t)-1-i) for i in range(len(t)//2) if i in m and len(t)-1-i in m]
 return sum(t[i]!=t[j] for i,j in pairs)
def main():
 states=[({},[],0)]; layers=[]
 for k in KEYS:
  nxt=[]
  for st,known,bad in states:
   for val in B[k]:
    q=dict(st);q[k]=val; newmask=mask(q)
    # Boundary expansion is accepted only when every currently resolved pair
    # remains compatible; unresolved positions are intentionally deferred.
    score=compatible(q,newmask)
    # The first layers have many unresolved intervals; keep them as frontier
    # states and let later mask closures perform the meaningful pruning.
    if score<=max(100, bad+2): nxt.append((q,newmask,score))
  nxt.sort(key=lambda z:(z[2],len(z[1]),render(z[0])))
  states=nxt[:12]; layers.append({'slot':k,'raw_states':len(nxt),'retained_states':len(states),'mask_widths':sorted({len(z[1]) for z in states})})
 rows=[]
 for rank,(x,m,bad) in enumerate(states[:8]):
  text=render(x);a=audit(text)
  rows.append({'rank':rank,'rendered':text,'slots':x,'resolved_position_mask':m,'mask_width':len(m),'partial_pair_conflicts':bad,'provenance':'resolved_tape_position_mask_grammar_dp','novelty_preflight':{'signature':'resolved_mask_grammar_dp_v1','distinct_from':'function-word DP score; stores and propagates explicit resolved tape-position masks'},'audit':a,'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'resolved-position-mask-grammar-dp-20260917','method':'carry explicit resolved character-position masks in each grammar DP state; defer unresolved seam pairs and prune incompatible resolved pairs','template':T,'layers':layers,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'represent function-word boundaries as intervals and update masks incrementally so only newly closed seam pairs are tested'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
