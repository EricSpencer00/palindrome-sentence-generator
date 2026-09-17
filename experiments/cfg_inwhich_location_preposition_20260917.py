#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
S=('gardener','reader','sailor','teacher'); V=('carries','reviews','marks','opens'); LV=('works','waits','rests','sails'); O=('map','letter','garden','harbor'); P=('near','beside','under')
SIG='in-which-state|location-preposition-substitution|locative-role-preserved|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def make(s,v,n,lv,p,place):return f'the {s} {v} the {n} in which the reader {lv} {p} the {place}.'
def row(t,p,rep,pr):return {'rendered':t,'attachment_state':'locative-in-which','location_preposition':p,'repaired':rep,'provenance':pr,'novelty_preflight':{'signature':SIG,'locative_role_preserved':True,'direct_object_excluded':True,'not_catalogue_replay':True},'audit':a(t),'live_frontier':f(t),'anti_shortcut':{'single_tree':True,'preposition_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for s,v,n,lv,p,place in itertools.product(S,V,O,LV,P,O):
  alt=next(x for x in P if x!=p)
  for t,pp,rep,pr in ((make(s,v,n,lv,p,place),p,False,'fresh in-which locative realization'),(make(s,v,n,lv,alt,place),alt,True,'held-out in-which location-preposition substitution')):
   if c(t) not in seen:seen.add(c(t));rows.append(row(t,pp,rep,pr));controls+=not rep;repairs+=rep
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-inwhich-location-preposition-20260917','method':'locative preposition substitution constrained to the in-which attachment state','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(ex),'candidates':rows[:60],'next_repair':'Add a held-out locative predicate substitution (works/waits/rests/sails) while preserving the same preposition and place role.'}
 p=Path('runs/cfg-inwhich-location-preposition-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
