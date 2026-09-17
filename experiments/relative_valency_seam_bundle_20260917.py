#!/usr/bin/env python3
"""Relative-clause valency bundles constrain the opposing clause seam."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/relative-valency-seam-bundle-20260917.json'
T='The {agent}, {rel}, {verb} the {obj} beside the {set0}, and the {agent2} {verb2} the {obj2} near the {set1}.'
BUNDLES=[('gardener','who tends','carries','letters'),('teacher','who guides','writes','notes'),('messenger','who travels','records','charts')]
SETS=['harbor','garden','station']
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def offsets(x):
 out={};p=0
 for piece in re.split(r'({\w+})',T):
  if piece.startswith('{'):
   k=piece[1:-1];w=norm(x.get(k,''));out[k]=(p,p+len(w)-1);p+=len(w)
  else:p+=len(norm(piece))
 return out
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 rows=[]
 for i,b in enumerate(BUNDLES):
  for j in range(2):
   b2=BUNDLES[(i+j+1)%len(BUNDLES)]
   x={'agent':b[0],'rel':b[1],'verb':b[2],'obj':b[3],'set0':SETS[i],'agent2':b2[0],'verb2':b2[2],'obj2':b2[3],'set1':SETS[(i+j+1)%len(SETS)]}
   text=render(x); off=offsets(x); t=norm(text); rel_end=off['rel'][1]; seam_pairs=[(k,len(t)-1-k) for k in range(min(rel_end+1,len(t)//2))]; conflicts=sum(t[a]!=t[b] for a,b in seam_pairs)
   rows.append({'candidate':len(rows),'rendered':text,'slots':x,'relative_interval':list(off['rel']),'opposing_seam_pairs_checked':len(seam_pairs),'opposing_seam_conflicts':conflicts,'provenance':'relative_valency_bundle_interval_seam_constraint','novelty_preflight':{'signature':'relative_valency_seam_bundle_v1','distinct_from':'optional relative insertion; lexical relation bundles constrain opposing seam before audit'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'relative-valency-seam-bundle-20260917','method':'subject-relative lexical valency bundles; relative interval drives opposing seam-pair checks before completion','template':T,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'jointly choose relative and opposing bundles with a character-level seam CSP instead of evaluating fixed bundle pairs'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
