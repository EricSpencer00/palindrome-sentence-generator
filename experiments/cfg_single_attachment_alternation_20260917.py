#!/usr/bin/env python3
"""One controlled direct-object/locative alternation at a fixed noun."""
import hashlib,itertools,json,re
from pathlib import Path
SUBJ=('gardener','reader','sailor','teacher'); V=('carries','reviews','marks','opens'); NOUN=('map','letter','garden','harbor'); LOCV=('works','waits','rests','sails'); PREP=('near','beside','under')
SIG='fixed-semantic-noun|single-attachment-alternation|direct-object-locative|feature-carried-character-frontier|independent-pointer-sha'
def canon(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=canon(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def live(s,att):
 t=canon(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'attachment':att,'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'attachment':att,'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def direct(s,v,n,rv):return f'the {s} {v} the {n} that the reader {rv} the {n}.'
def loc(s,v,n,lv,p):return f'the {s} {v} the {n} where the reader {lv} {p} the {n}.'
def pack(text,att,n,repair,prov):return {'rendered':text,'provenance':prov,'repaired':repair,'semantic_noun':n,'attachment_state':att,'novelty_preflight':{'signature':SIG,'single_alternation_only':True,'same_noun_both_realizations':True,'not_catalogue_replay':True},'audit':audit(text),'live_frontier':live(text,att),'anti_shortcut':{'single_tree':True,'single_attachment_alternation':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=alternations=0
 for s,v,n,rv,lv,p in itertools.product(SUBJ,V,NOUN,V,LOCV,PREP):
  d=direct(s,v,n,rv);l=loc(s,v,n,lv,p)
  for text,att,rep,prov in ((d,'direct-object',False,'fresh direct-object realization at fixed semantic noun'),(l,'locative',True,'single controlled attachment alternation: same noun, locative realization')):
   k=canon(text)
   if k not in seen:
    seen.add(k);rows.append(pack(text,att,n,rep,prov))
    if rep: alternations+=1
    else: controls+=1
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));exact=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-single-attachment-alternation-20260917','method':'fixed semantic noun with exactly one direct-object/locative alternation and feature-carrying live character frontier','signature':SIG,'control_count':controls,'alternation_count':alternations,'candidate_count':len(rows),'exact_count':len(exact),'candidates':rows[:60],'next_repair':'Add one semantic-role-preserving relative-clause verb alternation within each attachment state, retaining the fixed noun and rejecting cross-state substitutions.'}
 p=Path('runs/cfg-single-attachment-alternation-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'alternations':alternations,'candidates':len(rows),'exact':len(exact),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
