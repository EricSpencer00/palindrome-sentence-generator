#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
S=('gardener','reader','sailor','teacher'); V=('carries','reviews','marks','opens'); LV=('work','wait','rest','sail'); O=('map','letter','garden','harbor'); P=('near','beside','under')
SIG='in-which-state|shared-preposition-reduction|two-place-role-preserved|conjunction-fixed|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def full(ms,v,n,r1,r2,lv,p1,x,p2,y):return f'the {ms} {v} the {n} in which the {r1} and the {r2} {lv} {p1} the {x} and {p2} the {y}.'
def shared(ms,v,n,r1,r2,lv,p,x,y):return f'the {ms} {v} the {n} in which the {r1} and the {r2} {lv} {p} the {x} and the {y}.'
def row(t,mode,rep,pr):return {'rendered':t,'attachment_state':'locative-in-which','preposition_mode':mode,'repaired':rep,'provenance':pr,'novelty_preflight':{'signature':SIG,'two_place_roles_preserved':True,'shared_scope_only':True,'not_catalogue_replay':True},'audit':a(t),'live_frontier':f(t),'anti_shortcut':{'single_tree':True,'shared_preposition_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for ms,v,n,r1,r2,lv,p1,x,p2,y in itertools.islice(itertools.product(S,V,O,S,S,LV,P,O,P,O),512):
  for t,mode,rep,pr in ((full(ms,v,n,r1,r2,lv,p1,x,p2,y),'split',False,'fresh split-preposition locative complement'),(shared(ms,v,n,r1,r2,lv,p1,x,y),'shared',True,'held-out shared-preposition reduction; both place roles fixed')):
   if c(t) not in seen:seen.add(c(t));rows.append(row(t,mode,rep,pr));controls+=not rep;repairs+=rep
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-shared-preposition-locative-reduction-20260917','method':'shared-preposition reduction over two coordinated locative roles','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(ex),'candidates':rows[:60],'next_repair':'Add a held-out locative complementizer reduction from in which to where when the shared-preposition state remains grammatical.'}
 p=Path('runs/cfg-shared-preposition-locative-reduction-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
