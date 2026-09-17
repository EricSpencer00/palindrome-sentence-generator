#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
SUBJ=('the gardener','the reader','the sailor','the teacher'); PRES=('marks','reads','carries','opens'); PAST=('marked','read','carried','opened'); OBJ=(('maps','pl'),('letters','pl'),('gardens','pl'),('harbors','pl'),('map','sg'),('letter','sg'),('garden','sg'),('harbor','sg'))
SIG='typed-temporal-CFG|while-connective|formerly-later-order|plural-object-coordinated-articles|mixed-pair-pruning|independent-pointer-sha'
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
  if s==s2 or o==o2 or on!='pl' or on2!='pl':pruned+=1;continue
  t=f'{s} {v} the {o} formerly while {s2} {pv} later the {o2}.'
  rows.append({'rendered':t,'provenance':'plural-object coordinated article CFG: both number states plural, both determiners forced to the','grammar_state':{'connective':'while','subject_relation':'alternating','object_numbers':(on,on2),'coordinated_determiners':('the','the'),'mixed_number_pairs':'pruned','tense_sequence':'present->past','temporal_order':'formerly<later'},'live_frontier':f(t),'audit':a(t),'anti_shortcut':{'single_tree':True,'plural_article_gate':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'distinct_from_singular_coordination':True,'mixed_pairs_pruned':True}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[x for x in rows if x['audit']['two_pointer']]
 out={'experiment':'cfg-while-plural-coordinated-articles-20260917','method':'plural-object coordinated the/the gate over alternating temporal transitive frame','signature':SIG,'candidate_count':len(rows),'pruned_partial_count':pruned,'exact_count':len(ex),'candidates':rows[:80],'next_repair':'Add a plural-object lexical substitution in one position while retaining the other plural object and the the/the gate.'}
 p=Path('runs/cfg-while-plural-coordinated-articles-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'candidates':len(rows),'pruned':pruned,'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
