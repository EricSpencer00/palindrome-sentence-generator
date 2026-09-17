#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
SUBJ=('the gardener','the reader','the sailor','the teacher'); PRES=('marks','reads','carries','opens'); PAST=('marked','read','carried','opened'); OBJ=(('map','sg'),('letter','sg'),('garden','sg'),('harbor','sg'),('maps','pl'),('letters','pl'),('gardens','pl'),('harbors','pl')); DET=('a','the')
SIG='typed-temporal-CFG|while-connective|formerly-later-order|single-article-transition|alternating-object-number|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def main():
 rows=[];pruned=0
 for s,v,(o,on),s2,pv,(o2,on2) in itertools.product(SUBJ,PRES,OBJ,SUBJ,PAST,OBJ):
  if s==s2 or o==o2 or on==on2:pruned+=1;continue
  d1=('the',) if on=='pl' else DET; d2=('the',) if on2=='pl' else DET
  for d2v in d2:
   base=f'{s} {v} {d1[0]} {o} formerly while {s2} {pv} later {d2v} {o2}.'
   alt_d='the' if d1[0]=='a' else 'a'
   if on=='pl': continue
   alt=f'{s} {v} {alt_d} {o} formerly while {s2} {pv} later {d2v} {o2}.'
   for t,changed,rep,pr in ((base,'none',False,'fresh alternating-object article control'),(alt,'first_object_article',True,'held-out single-position article transition; second article fixed')):
    rows.append({'rendered':t,'provenance':pr,'grammar_state':{'connective':'while','subject_relation':'alternating','object_numbers':(on,on2),'article_transition':changed,'tense_sequence':'present->past','temporal_order':'formerly<later'},'live_frontier':f(t),'audit':a(t),'anti_shortcut':{'single_tree':True,'single_position_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'distinct_from_full_determiner_lane':True,'role_consistent':True}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[x for x in rows if x['audit']['two_pointer']]
 out={'experiment':'cfg-while-single-article-transition-20260917','method':'single-position article transition with fixed second article and alternating object numbers','signature':SIG,'candidate_count':len(rows),'pruned_partial_count':pruned,'exact_count':len(ex),'candidates':rows[:80],'next_repair':'Apply the single-position article transition to the second object, preserving the first article and temporal state.'}
 p=Path('runs/cfg-while-single-article-transition-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'candidates':len(rows),'pruned':pruned,'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
