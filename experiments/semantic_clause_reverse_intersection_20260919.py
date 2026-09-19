#!/usr/bin/env python3
"""Constructive semantic clause reverse-intersection pilot.

Generates complete clauses from typed subject/verb/object/place slots, indexes
normalized character tapes, and only then intersects each tape with its reverse.
No finished palindrome is copied or wrapped.
"""
import itertools, re, hashlib, json
from pathlib import Path
D={"det":["a","the"],"subj":["pilot","poet","scribe","sailor","keeper","baker","captain","gardener"],"verb":["marks","keeps","writes","carries","guards","sees","reads","mends"],"obj":["map","letter","chart","notes","swan","boat","book","garden"],"prep":["at","by","near","in"],"place":["dawn","shore","home","sea","spring","harbor"]}
T=(lambda d:f"{d['det']} {d['subj']} {d['verb']} {d['det']} {d['obj']} {d['prep']} {d['det']} {d['place']}.",lambda d:f"{d['det']} {d['subj']} {d['verb']} {d['det']} {d['obj']}.",lambda d:f"{d['subj']} {d['verb']} {d['obj']} {d['prep']} {d['place']}.")
def letters(s): return ''.join(c.lower() for c in s if c.isalpha())
def audit(s):
 t=letters(s); return {"letters":len(t),"exact":bool(t) and t==t[::-1],"sha256":hashlib.sha256(t.encode()).hexdigest(),"words":s[:-1].split()}
def main():
 clauses=[]
 for f in T:
  for vals in itertools.product(*D.values()):
   s=f(dict(zip(D,vals)))
   if len(letters(s))>=12: clauses.append(s)
 idx={}
 for s in clauses: idx.setdefault(letters(s),[]).append(s)
 pairs=[]
 for s in clauses:
  for r in idx.get(letters(s)[::-1],[]):
   if s!=r: pairs.append((s,r))
 result={"method":"typed complete semantic clauses indexed by exact reverse character tape","clauses":len(clauses),"distinct_tapes":len(idx),"reverse_pairs":len(pairs),"candidates":[{"rendered":a+' '+b,"audit":audit(a+' '+b),"provenance":{"left":a,"right":b,"generated":True,"catalogue":False}} for a,b in pairs[:20]],"positive_control":{"rendered":"An aide rips nine memos; some men inspire Diana.","audit":audit("An aide rips nine memos; some men inspire Diana.")},"next_repair":"add agreement-compatible auxiliaries and inflectional boundary variants, then intersect character obligations online"}
 Path('runs/semantic-clause-reverse-intersection-20260919.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({k:result[k] for k in ('clauses','distinct_tapes','reverse_pairs')}))
if __name__=='__main__': main()
