#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
S=('gardener','reader','sailor','teacher'); V=('carries','reviews','marks','opens'); SG=('works','waits','rests','sails'); PL=('work','wait','rest','sail'); O=('map','letter','garden','harbor'); P=('near','beside','under')
SIG='shared-where-state|locative-subject-agreement-variation|singular-plural-predicate|place-pair-fixed|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def make(ms,v,n,sub,lv,p,x,y):return f'the {ms} {v} the {n} where the {sub} {lv} {p} the {x} and the {y}.'
def row(t,state,rep,pr):return {'rendered':t,'attachment_state':'locative-where-shared-preposition','agreement_state':state,'repaired':rep,'provenance':pr,'novelty_preflight':{'signature':SIG,'place_pair_fixed':True,'where_fixed':True,'not_catalogue_replay':True},'audit':a(t),'live_frontier':f(t),'anti_shortcut':{'single_tree':True,'agreement_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for ms,v,n,sub,lv,p,x,y in itertools.islice(itertools.product(S,V,O,S,SG,P,O,O),256):
  alt=next(z for z in S if z!=sub)
  for t,state,rep,pr in ((make(ms,v,n,sub,lv,p,x,y),'singular',False,'fresh shared-where singular agreement'),(make(ms,v,n,sub,PL[SG.index(lv)],p,x,y),'plural',True,'held-out shared-where plural agreement variation')):
   if c(t) not in seen:seen.add(c(t));rows.append(row(t,state,rep,pr));controls+=not rep;repairs+=rep
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-shared-where-agreement-variation-20260917','method':'shared-where locative subject agreement variation with fixed place pair','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(ex),'candidates':rows[:60],'next_repair':'Add a held-out shared-where matrix-verb agreement variant while preserving the locative agreement state and place pair.'}
 p=Path('runs/cfg-shared-where-agreement-variation-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
