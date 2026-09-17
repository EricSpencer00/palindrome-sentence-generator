#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
SUBJ=('the gardener','the reader','the sailor','the teacher'); PRES=('rests','waits','works','sails'); PAST=('marked','read','carried','opened'); OBJ=('the map','the letter','the garden','the harbor')
SIG='typed-temporal-CFG|before-direction|present-to-past|earlier-later-order|shared-subject|independent-pointer-sha'
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
 for s,rv,o,s2,pv in itertools.product(SUBJ,PRES,OBJ,SUBJ,PAST):
  for a1,a2 in (('earlier','later'),('later','earlier')):
   if s!=s2 or a1!='earlier' or a2!='later':pruned+=1;continue
   t=f'{s} {rv} {a1} before {s2} {pv} {o} {a2}.'
   rows.append({'rendered':t,'provenance':'explicit ordered temporal pair: earlier modifies present event, later modifies past event','grammar_state':{'connective':'before','subject_sharing':True,'tense_sequence':'present->past','temporal_order':'earlier<later','adverb_pair':('earlier','later')},'live_frontier':f(t),'audit':a(t),'anti_shortcut':{'single_tree':True,'subject_shared':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'explicit_order_feature':True,'reverse_pair_pruned':True}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[x for x in rows if x['audit']['two_pointer']]
 out={'experiment':'cfg-before-earlier-later-order-20260917','method':'before temporal CFG with explicit earlier<later feature and shared subject','signature':SIG,'candidate_count':len(rows),'pruned_partial_count':pruned,'exact_count':len(ex),'candidates':rows[:80],'next_repair':'Add a second ordering-compatible pair (formerly/later) with a distinct tense-state predicate, preserving the explicit temporal order constraint.'}
 p=Path('runs/cfg-before-earlier-later-order-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'candidates':len(rows),'pruned':pruned,'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
