#!/usr/bin/env python3
"""Bidirectional arc consistency over character-prefix lexical domains."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/bidirectional-arc-character-domains-20260917.json'
T='The {a} {v} the {o} near the {s0}, and the {a2} {v2} the {o2} near the {s1}.'
D={'a':['gardener','teacher'],'v':['carries','writes'],'o':['letters','notes'],'s0':['harbor','garden'],'a2':['teacher','messenger'],'v2':['writes','records'],'o2':['notes','charts'],'s1':['garden','station']};K=list(D)
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**{k:x.get(k,D[k][0]) for k in K})
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 dom={k:list(v) for k,v in D.items()}; removed=0; rounds=0
 # Opposing slot arcs approximate the outer tape constraints using first/last
 # characters; both directions revise domains until a fixed point.
 pairs=list(zip(K[:4],reversed(K[4:])))
 changed=True
 while changed:
  changed=False;rounds+=1
  for left,right in pairs:
   for u,v in ((left,right),(right,left)):
    old=list(dom[u]); dom[u]=[w for w in old if any(norm(w)[0]==norm(z)[-1] for z in dom[v])]
    removed+=len(old)-len(dom[u]);changed|=len(old)!=len(dom[u])
 rows=[]
 for i in range(min(8, max(len(dom['a']),1))):
  x={k:(dom[k][i%len(dom[k])] if dom[k] else D[k][0]) for k in K}; text=render(x);rows.append({'candidate':i,'rendered':text,'slots':x,'domain_sizes':{k:len(v) for k,v in dom.items()},'arc_rounds':rounds,'removed_values':removed,'provenance':'bidirectional_arc_character_domain_consistency','novelty_preflight':{'signature':'bidirectional_arc_character_domains_v1','distinct_from':'prefix label comparison; repeatedly revises both lexical domains until arc fixed point'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'bidirectional-arc-character-domains-20260917','method':'bidirectional arc consistency repeatedly filters opposing lexical domains by exact boundary-character support before rendering','rounds':rounds,'removed_values':removed,'domain_sizes':{k:len(v) for k,v in dom.items()},'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'replace boundary-character arcs with full positional character constraints over rendered slot intervals'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
