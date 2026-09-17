#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
SUBJ=('the gardener','the reader','the sailor','the teacher'); PAST=('marked','read','carried','opened'); PRES=('rests','waits','works','sails'); OBJ=('the map','the letter','the garden','the harbor'); TEMP=('after','before','when')
SIG='typed-temporal-CFG|shared-subject|mixed-tense|temporal-connective|pre-render-tense-filter|independent-pointer-sha'
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
 for s,v,o,temp,s2,rv in itertools.product(SUBJ,PAST,OBJ,TEMP,SUBJ,PRES):
  if s!=s2:pruned+=1;continue
  # Temporal chart permits past event -> present state for after/when;
  # before requires the reverse order and is pruned in this realization.
  if temp=='before':pruned+=1;continue
  t=f'{s} {v} {o} {temp} {s2} {rv}.'
  rows.append({'rendered':t,'provenance':'fresh typed temporal CFG: shared subject, past event, temporal connective, present state','grammar_state':{'connective':temp,'subject_sharing':True,'tense_sequence':'past->present','roles':['event','state']},'live_frontier':f(t),'audit':a(t),'anti_shortcut':{'single_tree':True,'subject_shared':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'not_prior_causal_only':True,'temporal_mixed_tense':True,'pruned_before_render':True}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[x for x in rows if x['audit']['two_pointer']]
 out={'experiment':'cfg-causal-mixed-tense-temporal-20260917','method':'typed mixed-tense temporal CFG with shared subject and connective-specific pre-render filter','signature':SIG,'candidate_count':len(rows),'pruned_partial_count':pruned,'exact_count':len(ex),'candidates':rows[:80],'next_repair':'Add the reverse present-to-past temporal direction for before, with an explicit tense-order state and the same shared-subject obligation.'}
 p=Path('runs/cfg-causal-mixed-tense-temporal-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'candidates':len(rows),'pruned':pruned,'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
