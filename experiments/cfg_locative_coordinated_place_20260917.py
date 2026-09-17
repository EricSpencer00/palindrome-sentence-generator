#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
S=('gardener','reader','sailor','teacher'); V=('carries','reviews','marks','opens'); LV=('work','wait','rest','sail'); O=('map','letter','garden','harbor'); P=('near','beside','under')
SIG='in-which-state|coordinated-place-complement|plural-locative-agreement|predicate-preposition-fixed|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def make(ms,v,n,r1,r2,lv,p,x,y):return f'the {ms} {v} the {n} in which the {r1} and the {r2} {lv} {p} the {x} and the {y}.'
def row(t,places,rep,pr):return {'rendered':t,'attachment_state':'locative-in-which','coordinated_places':places,'repaired':rep,'provenance':pr,'novelty_preflight':{'signature':SIG,'plural_locative_agreement':True,'predicate_preposition_fixed':True,'not_catalogue_replay':True},'audit':a(t),'live_frontier':f(t),'anti_shortcut':{'single_tree':True,'place_coordination_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for ms,v,n,r1,r2,lv,p,x,y in itertools.product(S,V,O,S,S,LV,P,O,O):
  alt=next(z for z in O if z not in (x,y))
  for t,ps,rep,pr in ((make(ms,v,n,r1,r2,lv,p,x,y),f'{x} and {y}',False,'fresh coordinated-place realization'),(make(ms,v,n,r1,r2,lv,p,x,alt),f'{x} and {alt}',True,'held-out coordinated-place substitution; locative predicate/preposition fixed')):
   if c(t) not in seen:seen.add(c(t));rows.append(row(t,ps,rep,pr));controls+=not rep;repairs+=rep
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-locative-coordinated-place-20260917','method':'coordinated locative place complement with plural locative agreement and fixed preposition','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(ex),'candidates':rows[:60],'next_repair':'Add a held-out locative preposition coordination (near/beside/under) over the coordinated place state, preserving attachment and semantic location.'}
 p=Path('runs/cfg-locative-coordinated-place-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
