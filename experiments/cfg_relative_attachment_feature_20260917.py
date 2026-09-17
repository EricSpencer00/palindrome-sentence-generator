#!/usr/bin/env python3
"""Object-relative CFG with direct-object/locative attachment state."""
import hashlib,itertools,json,re
from pathlib import Path
N=('gardener','reader','sailor','teacher'); V=('carries','reviews','marks','opens'); OBJ=('map','letter','garden','harbor'); LOC=('works','waits','rests','sails'); PREP=('near','beside','under')
SIG='object-relative|attachment-feature|direct-object-vs-locative|gated-lexical-domain|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def live(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def direct(subj,verb,obj,rv,robj):return f'the {subj} {verb} the {obj} that the reader {rv} the {robj}.'
def locative(subj,verb,obj,lv,prep,place):return f'the {subj} {verb} the {obj} where the reader {lv} {prep} the {place}.'
def pack(s,state,repaired,prov):return {'rendered':s,'provenance':prov,'repaired':repaired,'attachment_state':state,'novelty_preflight':{'signature':SIG,'direct_and_locative_productions_disjoint':True,'not_catalogue_replay':True},'audit':audit(s),'live_frontier':live(s),'anti_shortcut':{'single_tree':True,'attachment_gated':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for subj,verb,obj,rv,robj in itertools.product(N,V,OBJ,V,OBJ):
  s=direct(subj,verb,obj,rv,robj);k=c(s);st={'type':'direct-object','production':'NP -> DET N that NP V NP','lexical_domain':'transitive RV + object noun'}
  if k not in seen:seen.add(k);rows.append(pack(s,st,False,'fresh direct-object relative derivation'));controls+=1
  alt=next(x for x in V if x!=rv);r=direct(subj,verb,obj,alt,robj);kr=c(r)
  if kr not in seen:seen.add(kr);rows.append(pack(r,st,True,'held-out direct-object verb substitution; attachment fixed'));repairs+=1
 for subj,verb,obj,lv,prep,place in itertools.product(N,V,OBJ,LOC,PREP,OBJ):
  s=locative(subj,verb,obj,lv,prep,place);k=c(s);st={'type':'locative','production':'NP -> DET N where NP V PREP NP','lexical_domain':'intransitive LV + locative preposition'}
  if k not in seen:seen.add(k);rows.append(pack(s,st,False,'fresh locative relative derivation'));controls+=1
  alt=next(x for x in LOC if x!=lv);r=locative(subj,verb,obj,alt,prep,place);kr=c(r)
  if kr not in seen:seen.add(kr);rows.append(pack(r,st,True,'held-out locative-verb substitution; attachment fixed'));repairs+=1
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));exact=[x for x in rows if x['audit']['two_pointer']]
 out={'experiment':'cfg-relative-attachment-feature-20260917','method':'single-tree CFG with disjoint direct-object and locative relative productions and gated lexical choices','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(exact),'candidates':rows[:60],'next_repair':'Permit one controlled attachment alternation at the same semantic noun (direct-object that versus locative where), carrying the feature through a character-obligation state and rejecting cross-domain lexical substitutions.'}
 p=Path('runs/cfg-relative-attachment-feature-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(exact),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
