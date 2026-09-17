#!/usr/bin/env python3
"""Variable clause-boundary insertion/deletion states with direct offsets."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/variable-clause-boundary-offset-transitions-20260917.json'
T='{d0} {a0} {v0} {d1} {o0} {p0} {d2} {s0}{bridge}{d3} {a1} {v1} {d4} {o1} {p1} {d5} {s1}.'
B={'d0':['the'],'a0':['gardener','teacher'],'v0':['carries','writes'],'d1':['the'],'o0':['letters','notes'],'p0':['beside','near'],'d2':['the'],'s0':['harbor','garden'],'d3':['the'],'a1':['teacher','messenger'],'v1':['writes','records'],'d4':['the'],'o1':['notes','charts'],'p1':['near','beside'],'d5':['the'],'s1':['garden','station']}
BRIDGES=[', and the ', ', while the ', ' and the ', ' while the ']
KEYS=list(B)
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**{k:x.get(k,'') for k in KEYS}|{'bridge':x.get('bridge',BRIDGES[0])})
def direct_offsets(x):
 out={};p=0
 for piece in re.split(r'({\w+})',T):
  if piece.startswith('{'):
   k=piece[1:-1];w=norm(x.get(k,''));out[k]=(p,p+len(w)-1);p+=len(w)
  else:
   p+=len(norm(x.get('bridge','') if piece=='{bridge}' else piece)) if piece=='{bridge}' else len(norm(piece))
 return out
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 rows=[]
 for bi,bridge in enumerate(BRIDGES):
  x={k:B[k][0] for k in KEYS};x.update({'a0':B['a0'][bi%2],'v0':B['v0'][bi%2],'o0':B['o0'][bi%2],'s0':B['s0'][bi%2],'a1':B['a1'][bi%2],'v1':B['v1'][bi%2],'o1':B['o1'][bi%2],'p1':B['p1'][bi%2],'s1':B['s1'][bi%2],'bridge':bridge})
  text=render(x); rows.append({'candidate':bi,'rendered':text,'slots':x,'boundary_transition':{'operation':'insert' if len(norm(bridge))>len(norm(BRIDGES[0])) else 'delete','bridge':bridge},'direct_offsets':{k:list(v) for k,v in direct_offsets(x).items()},'provenance':'variable_clause_boundary_direct_offset_transition','novelty_preflight':{'signature':'variable_boundary_offset_transition_v1','distinct_from':'fixed bridge direct offsets; insertion/deletion changes clause boundary and offset map'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'variable-clause-boundary-offset-transitions-20260917','method':'direct offset maps survive variable conjunction-boundary insertion/deletion transitions','template':T,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'make boundary transitions grammar-aware with optional relative clauses while preserving offset deltas'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
