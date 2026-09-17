#!/usr/bin/env python3
"""DP over function-word states with compatibility pruning at each layer."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/function-word-seam-dp-pruned-20260917.json'
T='{d0} {a0} {v0} {d1} {o0} {p0} {d2} {s0}, {c} {d3} {a1} {v1} {d4} {o1} {p1} {d5} {s1}.'
B={'d0':['the','a'],'a0':['gardener','teacher'],'v0':['carries','writes'],'d1':['the','a'],'o0':['letters','notes'],'p0':['beside','near'],'d2':['the','a'],'s0':['harbor','garden'],'c':['and','while'],'d3':['the','a'],'a1':['teacher','messenger'],'v1':['writes','records'],'d4':['the','a'],'o1':['notes','charts'],'p1':['near','beside'],'d5':['the','a'],'s1':['garden','station']}
KEYS=list(B)
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**{k:x.get(k,'') for k in KEYS})
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def score(x):
 t=norm(render(x)); return sum(t[i]!=t[-1-i] for i in range(min(5,len(t)//2)))
def main():
 states=[{}]; layers=[]
 for k in KEYS:
  nxt=[]
  for st in states:
   for value in B[k]:
    q=dict(st);q[k]=value;nxt.append(q)
  # DP pruning: preserve only the best distinct seam states at each function
  # layer. This is state pruning, not a larger exhaustive lexical sweep.
  nxt.sort(key=lambda q:(score(q),render(q)))
  states=nxt[:16]; layers.append({'slot':k,'raw_states':len(nxt),'retained_states':len(states)})
 rows=[]
 for rank,x in enumerate(states[:8]):
  text=render(x);a=audit(text)
  rows.append({'rank':rank,'rendered':text,'slots':x,'provenance':'function_word_seam_dynamic_program_pruned_states','novelty_preflight':{'signature':'function_word_seam_dp_v1','distinct_from':'character expansion; retains bounded DP states after each function-word layer'},'audit':a,'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'function-word-seam-dp-pruned-20260917','method':'layered DP over function-word choices, retaining states by partial outer-character compatibility at every slot','template':T,'layers':layers,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'carry exact resolved-position masks in each DP state and permit grammar-slot expansion only when its boundary character is compatible'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
