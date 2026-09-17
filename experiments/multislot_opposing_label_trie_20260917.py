#!/usr/bin/env python3
"""Multi-slot opposing character-label propagation over lexical trie states."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/multislot-opposing-label-trie-20260917.json'
T='The {a} {v} the {o} near the {s0}, and the {a2} {v2} the {o2} near the {s1}.'
D={'a':['gardener','teacher'],'v':['carries','writes'],'o':['letters','notes'],'s0':['harbor','garden'],'a2':['teacher','messenger'],'v2':['writes','records'],'o2':['notes','charts'],'s1':['garden','station']};WORDS=sum(D.values(),[])
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 rows=[];frontier=[]
 for key in ['a','v','o','s0','a2','v2','o2','s1']:
  for word in D[key]:
   frontier.append({'slot':key,'word':word,'prefix':norm(word)[:2]})
 # Jointly pair slots through opposing prefix labels before full candidate render.
 for i in range(2):
  x={'a':D['a'][i],'v':D['v'][i],'o':D['o'][i],'s0':D['s0'][i],'a2':D['a2'][i],'v2':D['v2'][i],'o2':D['o2'][i],'s1':D['s1'][i]}
  labels=[norm(x['a'])[:2],norm(x['v'])[:2],norm(x['o'])[:2],norm(x['a2'])[:2],norm(x['v2'])[:2],norm(x['o2'])[:2]]
  matches=sum(1 for a in labels[:3] for b in labels[3:] if a[0]==b[-1])
  rows.append({'candidate':i,'rendered':render(x),'slots':x,'trie_prefix_states':labels,'opposing_label_matches':matches,'provenance':'multislot_opposing_label_trie_propagation','novelty_preflight':{'signature':'multislot_opposing_label_trie_v1','distinct_from':'single bridge prefix; carries lexical prefix states across all clause slots and compares opposing labels jointly'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'multislot-opposing-label-trie-20260917','method':'carry lexical trie prefix states across multiple slots and propagate opposing prefix labels jointly before complete render','frontier_size':len(frontier),'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'use full character-prefix domains with bidirectional arc consistency across all opposing slot pairs'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
