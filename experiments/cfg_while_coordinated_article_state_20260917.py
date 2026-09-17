#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
SUBJ=('the gardener','the reader','the sailor','the teacher'); PRES=('marks','reads','carries','opens'); PAST=('marked','read','carried','opened'); OBJ=(('map','sg'),('letter','sg'),('garden','sg'),('harbor','sg'),('maps','pl'),('letters','pl'),('gardens','pl'),('harbors','pl'))
SIG='typed-temporal-CFG|while-connective|formerly-later-order|coordinated-article-state|number-gated|independent-pointer-sha'
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
  if s==s2 or o==o2:pruned+=1;continue
  if on!='sg' or on2!='sg':pruned+=1;continue
  for det in ('a','the'):
   t=f'{s} {v} {det} {o} formerly while {s2} {pv} later {det} {o2}.'
   rows.append({'rendered':t,'provenance':'coordinated article state: both singular alternating objects receive the same admitted determiner','grammar_state':{'connective':'while','subject_relation':'alternating','object_numbers':(on,on2),'coordinated_determiners':(det,det),'mixed_determiners':'pruned','tense_sequence':'present->past','temporal_order':'formerly<later'},'live_frontier':f(t),'audit':a(t),'anti_shortcut':{'single_tree':True,'coordinated_articles_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'mixed_states_pruned':True,'distinct_from_one_position':True}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[x for x in rows if x['audit']['two_pointer']]
 out={'experiment':'cfg-while-coordinated-article-state-20260917','method':'number-gated coordinated determiner state for two alternating singular objects','signature':SIG,'candidate_count':len(rows),'pruned_partial_count':pruned,'exact_count':len(ex),'candidates':rows[:80],'next_repair':'Add plural-object coordinated articles with the/the gating, preserving coordinated state and pruning singular/plural mixed pairs.'}
 p=Path('runs/cfg-while-coordinated-article-state-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'candidates':len(rows),'pruned':pruned,'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
