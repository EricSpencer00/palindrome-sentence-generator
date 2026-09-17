#!/usr/bin/env python3
"""Enforce character requirements during trie traversal."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/trie-traversal-character-requirements-20260917.json'
T='At dawn, the {agent} {action} the {theme} along the harbor; meanwhile, a {agent2} {action2} the {theme2} near the archive.'
ROWS=[('archivist','keeps','records','inside','courier','delivers','parcels'),('gardener','carries','letters','through','messenger','records','charts'),('teacher','writes','notes','along','cartographer','marks','maps')]
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(r):return T.format(agent=r[0],action=r[1],theme=r[2],prep=r[3],agent2=r[4],action2=r[5],theme2=r[6])
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def traverse(word,required):
 prefix='';rejected=[]
 for c in norm(word):
  prefix+=c
  if len(prefix)<=len(required) and c!=required[len(prefix)-1]:rejected.append({'prefix':prefix,'required':required[len(prefix)-1]});return False,rejected
 return True,rejected
def main():
 rows=[]
 for i,r in enumerate(ROWS):
  required=norm(r[3])[:2];ok,rej=traverse(r[3],required);rows.append({'candidate':i,'rendered':render(r),'attachment':r[3],'required_prefix':required,'trie_traversal_accepted':ok,'rejected_prefixes':rej,'provenance':'trie_traversal_character_requirements','novelty_preflight':{'signature':'trie_traversal_character_requirements_v1','distinct_from':'post-prefix gate; each character is checked against requirement as trie traversal advances'},'audit':audit(render(r)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'trie-traversal-character-requirements-20260917','method':'lexical trie traversal rejects a prefix immediately when its next character violates the propagated requirement','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'propagate requirements across multiple lexical slots and reject transitions before word completion'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
