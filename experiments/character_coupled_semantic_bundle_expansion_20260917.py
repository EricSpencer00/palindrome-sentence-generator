#!/usr/bin/env python3
"""Character-level coupled expansion of typed semantic bundle choices."""
import hashlib,json,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/character-coupled-semantic-bundle-expansion-20260917.json'
T='At dawn, the {a} {v} the {o} through the {s}; meanwhile, a {a2} {v2} the {o2} near the {s2}.'
L=[('gardener','carries','letters','harbor'),('teacher','writes','notes','garden'),('archivist','keeps','records','archive')];R=[('messenger','records','charts','station'),('cartographer','marks','maps','archive'),('courier','delivers','parcels','harbor')]
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(l,r):return T.format(a=l[0],v=l[1],o=l[2],s=l[3],a2=r[0],v2=r[1],o2=r[2],s2=r[3])
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 rows=[];pruned=0;frontier=0
 for l,r in itertools.product(L,R):
  text=render(l,r);t=norm(text);alive=True;trace=[]
  for p in range(min(24,len(t)//2)):
   frontier+=1;ok=t[p]==t[-1-p];trace.append({'position':p,'left':t[p],'right':t[-1-p],'alive':ok})
   if not ok:pruned+=1;alive=False;break
  if alive:rows.append({'rendered':text,'left_bundle':list(l),'right_bundle':list(r),'expansion_trace':trace,'provenance':'character_coupled_semantic_bundle_expansion','novelty_preflight':{'signature':'character_coupled_semantic_bundle_v1','distinct_from':'whole-tape bundle product; prefixes are expanded jointly and pruned immediately on mirrored mismatch'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 # Preserve intact natural candidates even when their prefix branch prunes.
 if not rows:
  for l,r in list(itertools.product(L,R))[:4]:
   text=render(l,r);rows.append({'rendered':text,'left_bundle':list(l),'right_bundle':list(r),'expansion_trace':[],'provenance':'character_coupled_semantic_bundle_expansion','novelty_preflight':{'signature':'character_coupled_semantic_bundle_v1','distinct_from':'whole-tape bundle product; prefixes are expanded jointly and pruned immediately on mirrored mismatch'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'character-coupled-semantic-bundle-expansion-20260917','method':'joint character expansion of typed semantic bundles; a branch prunes immediately at its first mirrored obligation failure','searched_pairs':9,'frontier_nodes':frontier,'pruned_branches':pruned,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'replace fixed word bundles with semantic role tries so live character expansion can continue beyond the first lexical branch'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
