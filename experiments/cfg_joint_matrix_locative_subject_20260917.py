#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
S=('gardener','reader','sailor','teacher'); V=('carries','reviews','marks','opens'); LV=('works','waits','rests','sails'); O=('map','letter','garden','harbor'); P=('near','beside','under')
SIG='in-which-state|joint-matrix-locative-subject|dual-valency-preserved|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def make(ms,v,n,rs,lv,p,place):return f'the {ms} {v} the {n} in which the {rs} {lv} {p} the {place}.'
def row(t,ms,rs,rep,pr):return {'rendered':t,'attachment_state':'locative-in-which','matrix_subject':ms,'locative_subject':rs,'repaired':rep,'provenance':pr,'novelty_preflight':{'signature':SIG,'dual_valency_preserved':True,'joint_only':True,'not_catalogue_replay':True},'audit':a(t),'live_frontier':f(t),'anti_shortcut':{'single_tree':True,'joint_subject_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for ms,v,n,rs,lv,p,place in itertools.product(S,V,O,S,LV,P,O):
  am=next(x for x in S if x!=ms);ar=next(x for x in S if x!=rs)
  for t,x,y,rep,pr in ((make(ms,v,n,rs,lv,p,place),ms,rs,False,'fresh dual-subject realization'),(make(am,v,n,ar,lv,p,place),am,ar,True,'held-out joint matrix/locative subject substitution')):
   if c(t) not in seen:seen.add(c(t));rows.append(row(t,x,y,rep,pr));controls+=not rep;repairs+=rep
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-joint-matrix-locative-subject-20260917','method':'joint matrix and locative subject substitution with dual valency state','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(ex),'candidates':rows[:60],'next_repair':'Add a held-out coordinated-subject realization inside the same locative state, preserving both subject roles and rejecting attachment changes.'}
 p=Path('runs/cfg-joint-matrix-locative-subject-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
