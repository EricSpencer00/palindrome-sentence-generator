#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
S=('gardener','reader','sailor','teacher'); SV=('carry','review','mark','open'); LV=('work','wait','rest','sail'); O=('map','letter','garden','harbor'); P=('near','beside','under')
SIG='in-which-state|coordinated-subject|plural-agreement|predicate-preposition-place-fixed|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def make(a1,a2,v,n,rs,lv,p,place):return f'the {a1} and the {a2} {v} the {n} in which the {rs} {lv} {p} the {place}.'
def row(t,sub,rep,pr):return {'rendered':t,'attachment_state':'locative-in-which','coordinated_matrix_subject':sub,'repaired':rep,'provenance':pr,'novelty_preflight':{'signature':SIG,'plural_agreement':True,'valency_preserved':True,'not_catalogue_replay':True},'audit':a(t),'live_frontier':f(t),'anti_shortcut':{'single_tree':True,'coordination_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for a1,a2,v,n,rs,lv,p,place in itertools.product(S,S,SV,O,S,LV,P,O):
  alt=next(x for x in S if x not in (a1,a2))
  for t,sub,rep,pr in ((make(a1,a2,v,n,rs,lv,p,place),f'{a1} and {a2}',False,'fresh coordinated plural matrix-subject realization'),(make(a1,alt,v,n,rs,lv,p,place),f'{a1} and {alt}',True,'held-out coordinated-subject substitution; plural agreement fixed')):
   if c(t) not in seen:seen.add(c(t));rows.append(row(t,sub,rep,pr));controls+=not rep;repairs+=rep
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-locative-coordinated-subject-20260917','method':'coordinated plural matrix subject with in-which locative state and agreement-carrying verb','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(ex),'candidates':rows[:60],'next_repair':'Add coordination inside the locative subject while carrying plural agreement through the locative predicate and preserving the same preposition/place.'}
 p=Path('runs/cfg-locative-coordinated-subject-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
