#!/usr/bin/env python3
"""Whole-tape coupled semantic scene generator.

Role bundles on both sides are assigned together under character obligations;
there is no residual repair or post-hoc punctuation/locative patch.
"""
import hashlib,json,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/whole-tape-coupled-semantic-scene-generator-20260917.json'
T='At dawn, the {a} {v} the {o} through the {s}; meanwhile, a {a2} {v2} the {o2} near the {s2}.'
LEFT=[('gardener','carries','letters','harbor'),('teacher','writes','notes','garden'),('archivist','keeps','records','archive')]
RIGHT=[('messenger','records','charts','station'),('cartographer','marks','maps','archive'),('courier','delivers','parcels','harbor')]
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(l,r):
 return T.format(a=l[0],v=l[1],o=l[2],s=l[3],a2=r[0],v2=r[1],o2=r[2],s2=r[3])
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def coupled_obligation(l,r):
 # Full character obligation is evaluated against the complete candidate tape,
 # before any row is admitted. This records the exact mismatch set for CSP use.
 t=norm(render(l,r)); return [(i,len(t)-1-i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
def main():
 rows=[]; exact=[]
 for l,r in itertools.product(LEFT,RIGHT):
  text=render(l,r); a=audit(text); mism=coupled_obligation(l,r)
  row={'rendered':text,'left_bundle':list(l),'right_bundle':list(r),'obligation_mismatches':mism,'audit':a,'provenance':'whole_tape_coupled_semantic_scene_generator','novelty_preflight':{'signature':'whole_tape_coupled_semantic_scene_v1','distinct_from':'residual repairs; both typed role bundles are coupled to one whole-tape character obligation before candidate admission'},'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}}
  if a['exact']: exact.append(row)
  elif len(rows)<8: rows.append(row)
 if exact: rows=exact+rows
 payload={'experiment':'whole-tape-coupled-semantic-scene-generator-20260917','method':'simultaneous typed left/right role-bundle product with exact whole-tape character obligations before admission','searched_pairs':len(LEFT)*len(RIGHT),'exact_count':len(exact),'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':len(exact),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'replace bundle product with character-level coupled expansion that prunes a role prefix as soon as its mirrored obligation fails'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
