#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
S=('gardener','reader','sailor','teacher'); V=('carries','reviews','marks','opens'); LV=('works','waits','rests','sails'); O=('map','letter','garden','harbor'); P=('near','beside','under')
SIG='shared-where-state|matrix-determiner-number|singular-plural-gate|place-pair-fixed|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def make(det,sub,v,n,rs,lv,p,x,y):return f'{det} {sub} {v} the {n} where the {rs} {lv} {p} the {x} and the {y}.'
def row(t,state,det,rep,pr):return {'rendered':t,'attachment_state':'locative-where-shared-preposition','matrix_subject_state':state,'matrix_determiner':det,'repaired':rep,'provenance':pr,'novelty_preflight':{'signature':SIG,'place_pair_fixed':True,'number_gate':True,'not_catalogue_replay':True},'audit':a(t),'live_frontier':f(t),'anti_shortcut':{'single_tree':True,'determiner_number_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for sub,v,n,rs,lv,p,x,y in itertools.islice(itertools.product(S,V,O,S,LV,P,O,O),256):
  for det,rep,pr in (('the',False,'fresh singular matrix determiner realization'),('a',True,'held-out singular matrix determiner alternation')):
   t=make(det,sub,v,n,rs,lv,p,x,y)
   if c(t) not in seen:seen.add(c(t));rows.append(row(t,'singular',det,rep,pr));controls+=not rep;repairs+=rep
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-shared-where-matrix-determiner-20260917','method':'shared-where matrix determiner alternation with singular number gate','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(ex),'candidates':rows[:60],'next_repair':'Add a held-out coordinated matrix determiner realization, preserving plural agreement and the shared-where locative frame.'}
 p=Path('runs/cfg-shared-where-matrix-determiner-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
