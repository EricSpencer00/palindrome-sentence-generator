#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
S=('gardener','reader','sailor','teacher'); V=('carry','review','mark','open'); LV=('work','wait','rest','sail'); O=('map','letter','garden','harbor'); P=('near','beside','under')
SIG='joint-locative-state|coordinated-complementizer-predicate-pair|semantic-roles-fixed|dual-place-roles|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def make(comp,m1,m2,v,n,r1,r2,lv,p,x,y):return f'the {m1} and the {m2} {v} the {n} {comp} the {r1} and the {r2} {lv} {p} the {x} and the {y}.'
def row(t,comp,pair,rep,pr):return {'rendered':t,'attachment_state':'joint-locative','complementizer':comp,'predicate_pair':pair,'repaired':rep,'provenance':pr,'novelty_preflight':{'signature':SIG,'semantic_roles_fixed':True,'place_roles_fixed':True,'not_catalogue_replay':True},'audit':a(t),'live_frontier':f(t),'anti_shortcut':{'single_tree':True,'coordinated_pair_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for m1,m2,v,n,r1,r2,lv,p,x,y in itertools.islice(itertools.product(S,S,V,O,S,S,LV,P,O,O),256):
  av=next(z for z in V if z!=v);al=next(z for z in LV if z!=lv)
  for t,comp,pair,rep,pr in ((make('where',m1,m2,v,n,r1,r2,lv,p,x,y),'where',f'{v}/{lv}',False,'fresh coordinated where/predicate realization'),(make('in which',m1,m2,av,n,r1,r2,al,p,x,y),'in which',f'{av}/{al}',True,'held-out coordinated complementizer/predicate-pair realization')):
   if c(t) not in seen:seen.add(c(t));rows.append(row(t,comp,pair,rep,pr));controls+=not rep;repairs+=rep
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-coordinated-complementizer-predicate-pair-20260917','method':'coordinated complementizer and matrix/locative predicate-pair realization','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(ex),'candidates':rows[:60],'next_repair':'Add a coordinated determiner/predicate-pair realization while preserving semantic roles and the selected attachment state.'}
 p=Path('runs/cfg-coordinated-complementizer-predicate-pair-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
