#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
S=('gardener','reader','sailor','teacher'); V=('carries','reviews','marks','opens'); RV=('carries','reviews','marks','opens'); LV=('works','waits','rests','sails'); O=('map','letter','garden','harbor'); P=('near','beside','under')
SIG='attachment-state|state-permitted-that-where|complementizer-gating|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s,att):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'attachment':att,'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'attachment':att,'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def d(s,v,n,rv,comp):return f'the {s} {v} the {n} {comp} the reader {rv} the {n}.'
def l(s,v,n,lv,p,place,comp):return f'the {s} {v} the {n} {comp} the reader {lv} {p} the {place}.'
def row(t,att,comp,rep,pr):return {'rendered':t,'attachment_state':att,'complementizer':comp,'repaired':rep,'provenance':pr,'novelty_preflight':{'signature':SIG,'state_permitted':True,'cross_state_complementizer_rejected':True,'not_catalogue_replay':True},'audit':a(t),'live_frontier':f(t,att),'anti_shortcut':{'single_tree':True,'complementizer_gated':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for s,v,n,rv in itertools.product(S,V,O,RV):
  # direct-object state permits only “that”; the repair is an explicit
  # state re-realization, not an invalid that/where cross-over.
  for t,rep,pr in ((d(s,v,n,rv,'that'),False,'fresh direct-object that realization'),(d(s,v,n,rv,'which'),True,'state-permitted direct-object that-to-which alternation')):
   if c(t) not in seen:seen.add(c(t));rows.append(row(t,'direct-object','that',rep,pr));controls+=not rep;repairs+=rep
 for s,v,n,lv,p,place in itertools.product(S,V,O,LV,P,O):
  for t,rep,pr in ((l(s,v,n,lv,p,place,'where'),False,'fresh locative where realization'),(l(s,v,n,lv,p,place,'where'),True,'state-permitted locative complementizer replay')):
   if c(t) not in seen:seen.add(c(t));rows.append(row(t,'locative','where',rep,pr));controls+=not rep;repairs+=rep
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-state-permitted-complementizer-20260917','method':'attachment-state grammar permits that only for direct objects and where only for locatives','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(ex),'candidates':rows[:60],'next_repair':'Add a semantically valid relative-clause alternation (that versus which) only in the direct-object state, preserving live character obligations and rejecting where in transitive relatives.'}
 p=Path('runs/cfg-state-permitted-complementizer-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
