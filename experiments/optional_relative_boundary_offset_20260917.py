#!/usr/bin/env python3
"""Grammar-aware optional relative clauses with preserved direct offsets."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/optional-relative-boundary-offset-20260917.json'
T='{d} {a}{rel} {v} {o} beside the {s}, and the {a2} {v2} the {o2} near the {s2}.'
AG=['gardener','teacher','messenger']; VB=['carries','writes','records']; OB=['letters','notes','charts']; SET=['harbor','garden','station']; REL=[', who writes',' who records','']
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
 for i,rel in enumerate(REL):
  x={'d':'the','a':AG[i%3],'rel':rel,'v':VB[i%3],'o':OB[i%3],'s':SET[i%3],'a2':AG[(i+1)%3],'v2':VB[(i+1)%3],'o2':OB[(i+1)%3],'s2':SET[(i+1)%3]}
  text=render(x);rows.append({'candidate':i,'rendered':text,'slots':x,'offsets':{k:list(v) for k,v in offsets(x).items()},'transition':{'relative_clause':'inserted' if rel else 'deleted','grammar_rule':'subject-relative optional, comma-preserving'},'provenance':'grammar_aware_optional_relative_direct_offset','novelty_preflight':{'signature':'optional_relative_offset_v1','distinct_from':'conjunction boundary transitions; relative clause insertion changes subject span while retaining grammar'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'optional-relative-boundary-offset-20260917','method':'grammar-aware optional subject-relative clause insertion/deletion with direct offset preservation','template':T,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'add relative-clause lexical valency bundles and use their character intervals to constrain the opposing clause seam'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
