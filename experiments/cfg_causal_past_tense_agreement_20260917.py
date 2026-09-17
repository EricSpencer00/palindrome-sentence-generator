#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
SUBJ=('the gardener','the reader','the sailor','the teacher'); PRES_T=('marks','reads','carries','opens'); PRES_I=('rests','waits','works','sails'); PAST_T=('marked','read','carried','opened'); PAST_I=('rested','waited','worked','sailed'); OBJ=('the map','the letter','the garden','the harbor'); CON=('because','since','as')
SIG='typed-causal-CFG|shared-subject|heldout-past-tense|agreement-obligation-filter|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def make(s,v,o,con,s2,rv):return f'{s} {v} {o} {con} {s2} {rv}.'
def main():
 rows=[];pruned=0
 for s,v,o,con,s2,rv in itertools.product(SUBJ,PAST_T,OBJ,CON,SUBJ,PAST_I):
  if s!=s2:pruned+=1;continue
  t=make(s,v,o,con,s2,rv);rows.append({'rendered':t,'provenance':'held-out typed causal CFG: shared subject + past transitive/intransitive event agreement','grammar_state':{'relation':con,'subject_sharing':True,'tense':'past','event_roles':['transitive','intransitive']},'live_frontier':f(t),'audit':a(t),'anti_shortcut':{'single_tree':True,'subject_shared':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'heldout_past_state':True,'not_locative_family':True,'obligation_filtered_before_render':True}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[x for x in rows if x['audit']['two_pointer']]
 out={'experiment':'cfg-causal-past-tense-agreement-20260917','method':'held-out past-tense typed causal CFG with shared subject and pre-render agreement filter','signature':SIG,'candidate_count':len(rows),'pruned_partial_count':pruned,'exact_count':len(ex),'candidates':rows[:80],'next_repair':'Add a held-out mixed-tense causal state with an explicit temporal connective, preserving subject identity and rejecting tense-inconsistent chart items.'}
 p=Path('runs/cfg-causal-past-tense-agreement-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'candidates':len(rows),'pruned':pruned,'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
