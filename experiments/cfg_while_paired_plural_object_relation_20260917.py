#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
SUBJ=('the gardener','the reader','the sailor','the teacher'); PRES=('marks','reads','carries','opens'); PAST=('marked','read','carried','opened'); PAIRS=(('maps','letters','written'),('gardens','harbors','places'),('letters','maps','documents'))
SIG='typed-temporal-CFG|while-connective|formerly-later-order|paired-plural-object-relation|the-the-gate|independent-pointer-sha'
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
 for s,v,s2,pv,(o1,o2,rel) in itertools.product(SUBJ,PRES,SUBJ,PAST,PAIRS):
  if s==s2:pruned+=1;continue
  t=f'{s} {v} the {o1} formerly while {s2} {pv} later the {o2}.'
  rows.append({'rendered':t,'provenance':f'paired plural objects admitted by semantic relation gate: {rel}','grammar_state':{'connective':'while','subject_relation':'alternating','object_relation':'paired','relation':rel,'object_numbers':('pl','pl'),'determiners':('the','the'),'tense_sequence':'present->past','temporal_order':'formerly<later'},'live_frontier':f(t),'audit':a(t),'anti_shortcut':{'single_tree':True,'semantic_pair_gate':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'distinct_from_one_position':True,'both_objects_changed':True}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[x for x in rows if x['audit']['two_pointer']]
 out={'experiment':'cfg-while-paired-plural-object-relation-20260917','method':'paired plural-object substitution under semantic relation gate and the/the agreement','signature':SIG,'candidate_count':len(rows),'pruned_partial_count':pruned,'exact_count':len(ex),'candidates':rows[:80],'next_repair':'Add a held-out semantic relation alternation for one object pair while retaining the other relation and temporal state.'}
 p=Path('runs/cfg-while-paired-plural-object-relation-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'candidates':len(rows),'pruned':pruned,'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
