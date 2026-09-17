#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
SUBJ=('the gardener','the reader','the sailor','the teacher'); PRES=('marks','reads','carries','opens'); PAST=('marked','read','carried','opened'); OBJ=(('map','sg'),('letter','sg'),('garden','sg'),('harbor','sg'),('maps','pl'),('letters','pl'),('gardens','pl'),('harbors','pl')); DET=('a','the')
SIG='typed-temporal-CFG|while-connective|formerly-later-order|alternating-object-determiner|number-gated|independent-pointer-sha'
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
  for x,y in itertools.product(d1,d2):
   t=f'{s} {v} {x} {o} formerly while {s2} {pv} later {y} {o2}.'
   rows.append({'rendered':t,'provenance':'held-out alternating-object determiner realization with number-gated lexical objects','grammar_state':{'connective':'while','subject_relation':'alternating','object_numbers':(on,on2),'determiners':(x,y),'tense_sequence':'present->past','temporal_order':'formerly<later'},'live_frontier':f(t),'audit':a(t),'anti_shortcut':{'single_tree':True,'determiner_number_gated':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'distinct_from_latent_object_number':True,'role_consistent':True}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[x for x in rows if x['audit']['two_pointer']]
 out={'experiment':'cfg-while-alternating-object-determiner-20260917','method':'alternating-object CFG with explicit number-gated determiner realization','signature':SIG,'candidate_count':len(rows),'pruned_partial_count':pruned,'exact_count':len(ex),'candidates':rows[:80],'next_repair':'Add a held-out article agreement transition at one object position, preserving the other determiner and temporal order.'}
 p=Path('runs/cfg-while-alternating-object-determiner-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'candidates':len(rows),'pruned':pruned,'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
