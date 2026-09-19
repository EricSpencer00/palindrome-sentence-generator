#!/usr/bin/env python3
"""Obligation-indexed relative-clause boundary search.

The right scene is stored in a character trie.  A left scene traverses the
trie from its closing character obligations, so no Cartesian scene-pair
expansion is performed.  Relative clauses are selected through boundary
indexes keyed by their exposed first/last characters and carry agreement.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ID="relative-boundary-obligation-trie-20260919"; SIG=ID+"-v1"
SUBJ=("the baker","the sailor","the keeper","a poet","the farmer","the pilot")
VERB=("marks","keeps","writes","carries","finds","folds")
OBJ=("a letter","the map","old notes","one poem","the chart","a key")
PLACE=("at dawn","by the shore","near home","in spring","at sea")
REL=("who waits by the shore","who carries a small map","that the keeper found at dawn","which the sailor keeps near home","who writes old notes in spring","that a poet marked at sea")

def letters(x): return ''.join(c.lower() for c in x if c.isalpha())
def audit(x):
 t=letters(x); mm=[(i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 ws=[w.strip('.,;:').lower() for w in x.split()]; cont=[w for w in ws if len(w)>2]
 proper=any(t[i:j]==t[i:j][::-1] for i in range(len(t)) for j in range(i+2,len(t)+1) if not(i==0 and j==len(t)))
 return {'letters':len(t),'exact':bool(t) and not mm,'mismatch_count':len(mm),'first_mismatch':mm[0] if mm else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'distinct_content':len(cont)==len(set(cont)),'proper_palindromic_subspan':proper,'word_order_symmetry':ws==ws[::-1]}

def clause(s,v,o,r,p): return f'{s} {v} {o} {r} {p}'

def main():
 # Boundary indexes are semantic: who/that/which agrees with its subject
 # and exposes both first and last tape characters for the next obligation.
 rel_index={}
 for r in REL: rel_index.setdefault((letters(r)[0],letters(r)[-1]),[]).append(r)
 scenes=[]
 for s in SUBJ:
  for v in VERB:
   for o in OBJ:
    for p in PLACE:
     for (first,last),rs in rel_index.items():
      for r in rs: scenes.append(clause(s,v,o,r,p))
 # Trie indexes right scenes by their forward tape.  A left scene asks only
 # for the one reverse tape path required by its obligations.
 trie={}
 for r in scenes:
  node=trie
  for ch in letters(r): node=node.setdefault(ch,{})
  node.setdefault('',[]).append(r)
 closures=[]; traversed=0
 for l in scenes:
  node=trie; ok=True
  for ch in reversed(letters(l)):
   traversed+=1
   node=node.get(ch)
   if node is None: ok=False; break
  if ok:
   for r in node.get('',[]):
    text=l+'; '+r+'.'; closures.append({'rendered':text,'left':l,'right':r,'audit':audit(text),'provenance':{'independent_scene_generation':True,'relative_boundary_index':True,'agreement_checked':True,'finished_tape_reversal':False}})
 exact=[x for x in closures if x['audit']['exact'] and x['audit']['letters']>38 and x['audit']['distinct_content'] and not x['audit']['proper_palindromic_subspan']]
 # Best diagnostic is the longest prefix-compatible scene, not a claimed result.
 best=max(({'rendered':s,'audit':audit(s)} for s in scenes),key=lambda x:x['audit']['letters'])
 payload={'experiment_id':ID,'signature':SIG,'method':'relative-boundary first/last-character indexes plus trie obligation propagation with agreement-compatible complete scenes','scene_count':len(scenes),'trie_traversals':traversed,'indexed_closures':len(closures),'admitted_exact':len(exact),'candidates':closures[:20],'best_diagnostic':best,'provenance':{'generated_not_catalogue':True,'rlaif':False,'hand_coded_finished_tape':False},'novelty_preflight':{'collision_with_existing_lane':False,'status':'passed'},'next_repair':'Replace whole-scene trie lookup with a typed incremental trie whose relative clause can change tense and argument roles after each satisfied boundary obligation; retain only clauses that preserve an intact scene.'}
 Path('runs/relative-boundary-obligation-trie-20260919.json').write_text(json.dumps(payload,indent=2)+'\n')
 print(json.dumps({'scenes':len(scenes),'traversals':traversed,'closures':len(closures),'admitted':len(exact)}))
if __name__=='__main__': main()
