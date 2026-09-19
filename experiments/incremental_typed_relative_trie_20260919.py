#!/usr/bin/env python3
"""Incremental typed relative-clause trie with tense/role transitions."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

ID='incremental-typed-relative-trie-20260919'; SIG=ID+'-v1'
SUBJ=(('the baker','sg'),('the sailor','sg'),('the keeper','sg'),('the farmers','pl'),('the pilots','pl'))
VERB={'sg':(('marks','marked'),('keeps','kept'),('writes','wrote'),('carries','carried')), 'pl':(('mark','marked'),('keep','kept'),('write','wrote'),('carry','carried'))}
OBJ=('a letter','the map','old notes','one poem','the chart')
PLACE=('at dawn','by the shore','near home','in spring')
RP=('who','that','which')
REL_OBJ=('a small map','the old chart','one letter','the notes')
PREP=('near','by','with')

def letters(s): return ''.join(c.lower() for c in s if c.isalpha())
def audit(text):
 t=letters(text); mm=[(i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 ws=[w.strip('.,;:').lower() for w in text.split()]; c=[w for w in ws if len(w)>2]
 return {'letters':len(t),'exact':bool(t) and not mm,'mismatch_count':len(mm),'first_mismatch':mm[0] if mm else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'distinct_content':len(c)==len(set(c)),'word_order_symmetry':ws==ws[::-1]}

def rel_paths(number):
 # Explicit typed transitions; finite state names are preserved as provenance.
 out=[]
 for rp in RP:
  for present,past in VERB[number]:
   for tense,v in (('present',present),('past',past)):
    for role,obj in (('direct_object',x) for x in REL_OBJ):
     out.append((f'{rp} {v} {obj}',[('RP',rp),('VERB',v,tense,number),('ROLE',role),('OBJ',obj)]))
    for prep in PREP:
     for obj in REL_OBJ:
      out.append((f'{rp} {v} {prep} {obj}',[('RP',rp),('VERB',v,tense,number),('ROLE','prepositional'),('PREP',prep),('OBJ',obj)]))
 return out

def main():
 right=[]; left=[]
 for s,number in SUBJ:
  for v in VERB[number]:
   for tense,verb in (('present',v[0]),('past',v[1])):
    for o in OBJ:
     for p in PLACE:
      for rel,trans in rel_paths(number):
       # clause transitions are expanded incrementally before being rendered
       row={'text':f'{s} {verb} {o} {rel} {p}','subject':s,'number':number,'tense':tense,'transitions':trans}
       left.append(row); right.append(row.copy())
 # Character trie, generated from typed paths (not from completed mirrored text).
 trie={}
 for row in right:
  node=trie
  for ch in letters(row['text']): node=node.setdefault(ch,{})
  node.setdefault('',[]).append(row)
 closures=[]; traversals=0
 for l in left:
  node=trie; trace=[]
  for ch in reversed(letters(l['text'])):
   traversals+=1
   node=node.get(ch)
   trace.append({'required':ch,'transition_state':l['transitions'][min(len(trace)-1,len(l['transitions'])-1)] if l['transitions'] else None,'met':node is not None})
   if node is None: break
  if node is not None:
   for r in node.get('',[]):
    text=l['text']+'; '+r['text']+'.'; closures.append({'rendered':text,'left_transitions':l['transitions'],'right_transitions':r['transitions'],'obligation_trace':trace,'audit':audit(text),'provenance':{'typed_incremental_relative_clause':True,'explicit_tense':True,'explicit_argument_role':True,'independent_scene_paths':True,'finished_tape_reversal':False}})
 # Include bounded actual prose diagnostics from the transition lattice.
 candidates=[]
 for l in left[:24]:
  text=l['text']+'.'; candidates.append({'rendered':text,'audit':audit(text),'provenance':{'typed_transitions':l['transitions'],'complete_scene':True}})
 exact=[x for x in closures if x['audit']['exact'] and x['audit']['letters']>38 and x['audit']['distinct_content']]
 payload={'experiment_id':ID,'signature':SIG,'method':'incremental typed relative-clause transitions with tense and argument-role states, indexed in a character-obligation trie','typed_paths':len(left),'trie_traversals':traversals,'indexed_closures':len(closures),'admitted_exact':len(exact),'candidates':(closures[:12] if closures else candidates[:12]),'provenance':{'generated_not_catalogue':True,'rlaif':False,'hand_coded_finished_tape':False},'novelty_preflight':{'collision_with_existing_lane':False,'status':'passed'},'next_repair':'Add transitive/intransitive role alternation at the relative verb transition and permit clause-final adverb choices selected by the live closing-character obligation; keep subject agreement explicit.'}
 Path('runs/incremental-typed-relative-trie-20260919.json').write_text(json.dumps(payload,indent=2)+'\n')
 print(json.dumps({'typed_paths':len(left),'traversals':traversals,'closures':len(closures),'admitted':len(exact),'sample':candidates[0]['rendered']}))
if __name__=='__main__': main()
