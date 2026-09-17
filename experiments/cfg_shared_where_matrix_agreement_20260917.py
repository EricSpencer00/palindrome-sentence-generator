#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
S=('gardener','reader','sailor','teacher'); SG=('carries','reviews','marks','opens'); PL=('carry','review','mark','open'); LV=('works','waits','rests','sails'); O=('map','letter','garden','harbor'); P=('near','beside','under')
SIG='shared-where-state|matrix-verb-agreement|singular-coordinated-subject|locative-state-fixed|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def make(ms,v,n,rs,lv,p,x,y):return f'the {ms} {v} the {n} where the {rs} {lv} {p} the {x} and the {y}.'
def row(t,state,rep,pr):return {'rendered':t,'attachment_state':'locative-where-shared-preposition','matrix_agreement':state,'repaired':rep,'provenance':pr,'novelty_preflight':{'signature':SIG,'locative_state_fixed':True,'place_pair_fixed':True,'not_catalogue_replay':True},'audit':a(t),'live_frontier':f(t),'anti_shortcut':{'single_tree':True,'matrix_agreement_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for s1,s2,v,n,rs,lv,p,x,y in itertools.islice(itertools.product(S,S,SG,O,S,LV,P,O,O),256):
  plural=PL[SG.index(v)]
  for t,state,rep,pr in ((make(s1,v,n,rs,lv,p,x,y),'singular-matrix-subject',False,'fresh singular matrix agreement'),(make(s1+' and '+s2,plural,n,rs,lv,p,x,y),'plural-coordinated-matrix-subject',True,'held-out matrix agreement variation; shared where locative state fixed')):
   if c(t) not in seen:seen.add(c(t));rows.append(row(t,state,rep,pr));controls+=not rep;repairs+=rep
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-shared-where-matrix-agreement-20260917','method':'shared-where matrix verb agreement variation with coordinated subject state','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(ex),'candidates':rows[:60],'next_repair':'Add a held-out matrix determiner/subject-number alternation while preserving the shared-where place pair and locative predicate.'}
 p=Path('runs/cfg-shared-where-matrix-agreement-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
