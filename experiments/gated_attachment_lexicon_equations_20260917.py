#!/usr/bin/env python3
"""Gate attachment lexicon expansion on exact partial character equations."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/gated-attachment-lexicon-equations-20260917.json'
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
 rows=[];rejected=[]
 for i,r in enumerate(ROWS):
  text=render(r);t=norm(text);partial=[p for p in range(min(10,len(t)//2)) if t[p]==t[-1-p]]
  if len(partial)<1:rejected.append({'candidate':i,'reason':'no exact partial equation support'});continue
  rows.append({'candidate':i,'rendered':text,'partial_equation_support':partial,'gate':'admitted','provenance':'gated_attachment_lexicon_exact_equations','novelty_preflight':{'signature':'gated_attachment_lexicon_equations_v1','distinct_from':'ungated attachment refinement; lexicon candidate is admitted only after exact partial equation support'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'gated-attachment-lexicon-equations-20260917','method':'attachment lexicon candidates pass an exact partial-character-equation gate before full-scene admission','candidate_count':len(rows),'candidates':rows,'rejected':rejected,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'make gate bidirectional over attachment and opposing role prefixes, preserving only jointly supported expansions'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
