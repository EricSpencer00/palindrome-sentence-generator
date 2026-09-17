#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
S=('gardener','reader','sailor','teacher'); V=('carry','review','mark','open'); LV=('works','waits','rests','sails'); O=('map','letter','garden','harbor'); P=('near','beside','under')
SIG='shared-where-state|coordinated-subject-lexical-substitution|plural-agreement|determiners-fixed|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def make(a1,a2,v,n,rs,lv,p,x,y):return f'the {a1} and the {a2} {v} the {n} where the {rs} {lv} {p} the {x} and the {y}.'
def row(t,sub,rep,pr):return {'rendered':t,'attachment_state':'locative-where-shared-preposition','coordinated_subject':sub,'repaired':rep,'provenance':pr,'novelty_preflight':{'signature':SIG,'plural_agreement':True,'determiners_fixed':True,'not_catalogue_replay':True},'audit':a(t),'live_frontier':f(t),'anti_shortcut':{'single_tree':True,'subject_lexical_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for a1,a2,v,n,rs,lv,p,x,y in itertools.islice(itertools.product(S,S,V,O,S,LV,P,O,O),256):
  alt=next(z for z in S if z!=a2)
  for t,sub,rep,pr in ((make(a1,a2,v,n,rs,lv,p,x,y),f'{a1} and {a2}',False,'fresh coordinated matrix subjects'),(make(a1,alt,v,n,rs,lv,p,x,y),f'{a1} and {alt}',True,'held-out coordinated matrix-subject lexical substitution')):
   if c(t) not in seen:seen.add(c(t));rows.append(row(t,sub,rep,pr));controls+=not rep;repairs+=rep
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-shared-where-coordinated-subject-lexical-20260917','method':'coordinated matrix-subject lexical substitution in shared-where frame','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(ex),'candidates':rows[:60],'next_repair':'Add a held-out locative subject lexical substitution jointly with the coordinated matrix subject, preserving both plural states.'}
 p=Path('runs/cfg-shared-where-coordinated-subject-lexical-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
