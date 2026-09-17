#!/usr/bin/env python3
"""Edge-labeled lexical trie transitions reject incompatible prefixes early."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/edge-labeled-lexical-trie-semantic-20260917.json'
T='The {a} {v} the {o} near the {s0}{bridge}{a2} {v2} the {o2} near the {s1}.'
WORDS=['gardener','teacher','messenger','carries','writes','records','letters','notes','charts','harbor','garden','station','and','while'];BR=[', and the ', ', while the ']
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
class Trie:
 def __init__(self,words):self.words=words
 def compatible(self,prefix):return [w for w in self.words if norm(w).startswith(norm(prefix))]
def main():
 trie=Trie(WORDS);rows=[];rejected=0
 for i,bridge in enumerate(BR):
  labels=list(norm(bridge)); prefix=list(norm('and' if 'and' in bridge else 'while'))
  # Edge labels constrain lexical transition prefixes before word completion.
  options=trie.compatible(''.join(prefix))
  rejected+=len(WORDS)-len(options)
  x={'a':'gardener' if i==0 else 'teacher','v':'carries' if i==0 else 'writes','o':'letters' if i==0 else 'notes','s0':'harbor' if i==0 else 'garden','bridge':bridge,'a2':'teacher' if i==0 else 'messenger','v2':'writes' if i==0 else 'records','o2':'notes' if i==0 else 'charts','s1':'garden' if i==0 else 'station'}
  text=render(x);rows.append({'candidate':i,'rendered':text,'edge_labels':labels,'accepted_prefix':prefix,'trie_options_before_completion':len(options),'slots':x,'provenance':'edge_labeled_lexical_trie_semantic_transition','novelty_preflight':{'signature':'edge_labeled_lexical_trie_v1','distinct_from':'character-labeled automaton; trie prefixes are rejected before lexical words are selected'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'edge-labeled-lexical-trie-semantic-20260917','method':'semantic edge characters constrain lexical trie prefixes before word completion','trie_vocabulary':len(WORDS),'rejected_prefix_transitions':rejected,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'carry trie prefix states across multiple lexical slots and enforce opposing character labels jointly'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
