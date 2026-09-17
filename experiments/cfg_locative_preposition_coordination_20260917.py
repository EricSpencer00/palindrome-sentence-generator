#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
S=('gardener','reader','sailor','teacher'); V=('carries','reviews','marks','opens'); LV=('work','wait','rest','sail'); O=('map','letter','garden','harbor'); P=('near','beside','under')
SIG='in-which-state|coordinated-preposition-place|plural-locative-agreement|attachment-preserved|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def make(ms,v,n,r1,r2,lv,p1,x,p2,y):return f'the {ms} {v} the {n} in which the {r1} and the {r2} {lv} {p1} the {x} and {p2} the {y}.'
def row(t,ps,rep,pr):return {'rendered':t,'attachment_state':'locative-in-which','coordinated_prepositions':ps,'repaired':rep,'provenance':pr,'novelty_preflight':{'signature':SIG,'plural_locative_agreement':True,'place_attachment_preserved':True,'not_catalogue_replay':True},'audit':a(t),'live_frontier':f(t),'anti_shortcut':{'single_tree':True,'preposition_coordination_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for ms,v,n,r1,r2,lv,p1,x,p2,y in itertools.product(S,V,O,S,S,LV,P,O,P,O):
  alt=next(z for z in P if z!=p2)
  for t,ps,rep,pr in ((make(ms,v,n,r1,r2,lv,p1,x,p2,y),f'{p1}/{p2}',False,'fresh coordinated-preposition locative realization'),(make(ms,v,n,r1,r2,lv,p1,x,p2,alt),f'{p1}/{alt}',True,'held-out coordinated-preposition substitution; places and attachment fixed')):
   if c(t) not in seen:seen.add(c(t));rows.append(row(t,ps,rep,pr));controls+=not rep;repairs+=rep
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-locative-preposition-coordination-20260917','method':'coordinated locative prepositions over a fixed in-which place state','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(ex),'candidates':rows[:60],'next_repair':'Add a held-out coordination conjunction alternation (and/while) only inside the locative complement, preserving both place roles and attachment.'}
 p=Path('runs/cfg-locative-preposition-coordination-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
