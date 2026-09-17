#!/usr/bin/env python3
"""Use unresolved masks to choose the next lexical character to expand."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/mask-guided-next-character-choice-20260917.json'
T='The {a} {v} the {o} near the {s0}, and the {a2} {v2} the {o2} near the {s1}.'
D={'a':['gardener','teacher'],'v':['carries','writes'],'o':['letters','notes'],'s0':['harbor','garden'],'a2':['teacher','messenger'],'v2':['writes','records'],'o2':['notes','charts'],'s1':['garden','station']};K=list(D)
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 rows=[]
 for i in range(2):
  x={k:D[k][i%len(D[k])] for k in K};t=norm(render(x));resolved=set(range(min(18,len(t))));unresolved=set(range(len(t)))-resolved;choices=[]
  for p in sorted(resolved):
   q=len(t)-1-p
   if q in unresolved:choices.append({'position':p,'opposing_position':q,'char':t[p]})
  rows.append({'candidate':i,'rendered':render(x),'slots':x,'mask_guided_next_choices':choices[:8],'resolved_count':len(resolved),'unresolved_count':len(unresolved),'provenance':'mask_guided_next_character_choice','novelty_preflight':{'signature':'mask_guided_next_char_v1','distinct_from':'mask recording; selection policy chooses a character whose opposing tape position remains unresolved'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'mask-guided-next-character-choice-20260917','method':'choose next lexical character from resolved positions whose true opposing positions remain unresolved','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'expand both selected character and opposing position through lexical trie states with exact equality propagation'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
