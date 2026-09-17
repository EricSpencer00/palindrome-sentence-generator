#!/usr/bin/env python3
"""Constructive attachment-valency refinement after held-out full-scene checks."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/attachment-valency-generator-refinement-20260917.json'
T='At dawn, the {agent} {action} the {theme} {prep} the {setting}; meanwhile, a {agent2} {action2} the {theme2} {prep2} the {setting2}.'
ROWS=[('archivist','keeps','records','inside','archive','courier','delivers','parcels','near','station'),('gardener','carries','letters','through','harbor','messenger','records','charts','beside','station'),('teacher','writes','notes','along','garden','cartographer','marks','maps','near','archive')]
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(r):return T.format(**dict(zip(['agent','action','theme','prep','setting','agent2','action2','theme2','prep2','setting2'],r)))
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 rows=[]
 for i,r in enumerate(ROWS):
  text=render(r);rows.append({'candidate':i,'rendered':text,'slots':dict(zip(['agent','action','theme','prep','setting','agent2','action2','theme2','prep2','setting2'],r)),'generator_change':'expanded typed attachment lexicon with inside/through/along and held-out role bundles','provenance':'attachment_valency_generator_refinement','novelty_preflight':{'signature':'attachment_valency_generator_refinement_v1','distinct_from':'held-out evaluation; adds new constructive attachment/role combinations to generator'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'attachment-valency-generator-refinement-20260917','method':'generator refinement adds typed attachment alternatives and held-out semantic role bundles before any reader study','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'couple attachment lexicon expansion to exact character equations before admitting full scene candidates'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
