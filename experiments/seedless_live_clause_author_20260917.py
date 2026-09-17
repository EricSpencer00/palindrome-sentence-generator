#!/usr/bin/env python3
"""Seedless live clause author: semantic frames grow under tape obligations."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/seedless-live-clause-author-20260917.json'
FRAMES=[
 ('The {agent} {verb} {object} near the {place}', 'The {agent2} {verb2} {object2} by the {place2}',
  [('gardener','teacher'),('carries','writes'),('letters','notes'),('harbor','garden')]),
 ('A {agent} {verb} the {object} beside a {place}', 'A {agent2} {verb2} the {object2} beside a {place2}',
  [('quiet baker','kind teacher'),('sends','keeps'),('bread','notes'),('river','garden')]),
 ('Our {agent} {verb} {object} at dawn', 'Our {agent2} {verb2} {object2} at dusk',
  [('calm editor','young reader'),('revises','copies'),('prose','notes')]),
]
def letters(s):return ''.join(c.lower() for c in s if c.isalpha())
def audit(t):
 i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]:return {'exact':False,'length':len(t),'first_mismatch':(i,j,t[i],t[j]),'sha256':hashlib.sha256(t.encode()).hexdigest()}
  i+=1;j-=1
 return {'exact':True,'length':len(t),'first_mismatch':None,'sha256':hashlib.sha256(t.encode()).hexdigest()}
def main():
 rows=[]; longest=[]
 for template,template2,slots in FRAMES:
  left,right=template,template2
  for vals in __import__('itertools').product(*[v for v in slots]):
   names=['agent','verb','object','place'][:len(vals)]
   a=dict(zip(names,vals)); b=dict(zip([x+'2' for x in names],vals))
   try:l=left.format(**a,**b);r=right.format(**a,**b)
   except KeyError:continue
   text=l+'. '+r; tape=letters(text); rec={'text':text,'length':len(tape),'audit':audit(tape),'provenance':'seedless authored semantic frame; live tape audit','anti_shortcut':True}
   rows.append(rec)
   if len(tape)> (longest[0]['length'] if longest else 0):longest=[rec]
 payload={'experiment':'seedless-live-clause-author-20260917','method':'semantic clause frames with immediate tape audit; no seed wrapping, mirrored-half construction, or post-hoc resegmentation','rows':rows,'longest_complete_proposals':longest,'exact_survivors':[r for r in rows if r['audit']['exact']],'next_repair':'replace frame value pairing with a character-synchronous role trie so each newly authored character constrains the opposing clause before slot completion'}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({'rows':len(rows),'exact':len(payload['exact_survivors']),'longest':longest[0]['length'] if longest else 0}))
if __name__=='__main__':main()
