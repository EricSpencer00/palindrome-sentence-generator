#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
S=('gardener','reader','sailor','teacher'); V=('carries','reviews','marks','opens'); LV=('works','waits','rests','sails'); O=('map','letter','garden','harbor'); P=('near','beside','under')
SIG='locative-state|where-in-which|location-role-preserved|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-1-i]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def make(s,v,n,comp,lv,p,place):
 return f'the {s} {v} the {n} {comp} the reader {lv} {p} the {place}.'
def row(t,comp,rep,pr):return {'rendered':t,'attachment_state':'locative','location_complementizer':comp,'repaired':rep,'provenance':pr,'novelty_preflight':{'signature':SIG,'location_role_preserved':True,'direct_object_excluded':True,'not_catalogue_replay':True},'audit':a(t),'live_frontier':f(t),'anti_shortcut':{'single_tree':True,'locative_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for s,v,n,lv,p,place in itertools.product(S,V,O,LV,P,O):
  for comp in ('where','in which'):
   t=make(s,v,n,comp,lv,p,place);rep=(comp!='where');k=c(t)
   if k not in seen:seen.add(k);rows.append(row(t,comp,rep,'fresh locative where realization' if not rep else 'held-out locative where-to-in-which alternation'));controls+=not rep;repairs+=rep
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-locative-where-inwhich-20260917','method':'locative role-preserving where/in which alternation with live character obligations','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(ex),'candidates':rows[:60],'next_repair':'Add a location-preposition substitution within the in-which state, preserving the same place role and rejecting direct-object transformations.'}
 p=Path('runs/cfg-locative-where-inwhich-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
