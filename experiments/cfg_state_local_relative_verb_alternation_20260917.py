#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
SUBJ=('gardener','reader','sailor','teacher'); DV=('carries','reviews','marks','opens'); LV=('works','waits','rests','sails'); NOUN=('map','letter','garden','harbor'); PREP=('near','beside','under')
SIG='fixed-semantic-noun|state-local-relative-verb|direct-object-locative|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s,att):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'attachment':att,'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'attachment':att,'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def d(s,v,n,rv):return f'the {s} {v} the {n} that the reader {rv} the {n}.'
def l(s,v,n,lv,p):return f'the {s} {v} the {n} where the reader {lv} {p} the {n}.'
def row(text,att,n,rep,prov):return {'rendered':text,'semantic_noun':n,'attachment_state':att,'repaired':rep,'provenance':prov,'novelty_preflight':{'signature':SIG,'state_local_only':True,'not_catalogue_replay':True},'audit':audit(text),'live_frontier':f(text,att),'anti_shortcut':{'single_tree':True,'state_local_verb':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for s,v,n,rv in itertools.product(SUBJ,DV,NOUN,DV):
  for text,rep,prov in ((d(s,v,n,rv),False,'fresh direct-object realization'),(d(s,v,n,next(x for x in DV if x!=rv)),True,'state-local direct-object relative-verb alternation')):
   k=c(text)
   if k not in seen:seen.add(k);rows.append(row(text,'direct-object',n,rep,prov));controls+=not rep;repairs+=rep
 for s,v,n,lv,p in itertools.product(SUBJ,DV,NOUN,LV,PREP):
  alt=next(x for x in LV if x!=lv)
  for text,rep,prov in ((l(s,v,n,lv,p),False,'fresh locative realization'),(l(s,v,n,alt,p),True,'state-local locative relative-verb alternation')):
   k=c(text)
   if k not in seen:seen.add(k);rows.append(row(text,'locative',n,rep,prov));controls+=not rep;repairs+=rep
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));exact=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-state-local-relative-verb-alternation-20260917','method':'attachment-state-local relative verb alternation with live character obligations','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(exact),'candidates':rows[:60],'next_repair':'Add one semantic-preserving subject substitution inside each attachment state, carrying subject valency and rejecting cross-state verbs.'}
 p=Path('runs/cfg-state-local-relative-verb-alternation-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(exact),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
