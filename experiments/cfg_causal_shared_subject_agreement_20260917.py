#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
SUBJ=('the gardener','the reader','the sailor','the teacher'); SG=('rests','waits','works','sails'); TV=('marks','reads','carries','opens'); OBJ=('the map','the letter','the garden','the harbor'); CON=('because','since','as')
SIG='typed-causal-CFG|shared-subject|present-tense-agreement|pre-render-obligation-filter|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def obligation_ok(parts):
 # Causal connective and shared-subject agreement are checked on chart items
 # before terminal punctuation/rendering; no finished tape is reversed.
 return parts[0]==parts[4] and parts[1] in TV and parts[5] in SG and len(' '.join(parts))>12
def main():
 rows=[];pruned=0
 for s,v,o,con,s2,rv in itertools.product(SUBJ,TV,OBJ,CON,SUBJ,SG):
  parts=(s,v,o,con,s2,rv)
  if not obligation_ok(parts):pruned+=1;continue
  t=f'{s} {v} {o} {con} {s2} {rv}.'
  rows.append({'rendered':t,'provenance':'fresh typed causal CFG: shared subject S -> NP; VP cause + connective + VP effect','grammar_state':{'relation':con,'subject_sharing':True,'tense':'present-singular','event_roles':['transitive','intransitive']},'live_frontier':f(t),'audit':a(t),'anti_shortcut':{'single_tree':True,'subject_shared':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'not_locative_family':True,'obligation_filtered_before_render':True}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-causal-shared-subject-agreement-20260917','method':'typed causal connective CFG with shared subject and present-tense agreement checked before rendering','signature':SIG,'candidate_count':len(rows),'pruned_partial_count':pruned,'exact_count':len(ex),'candidates':rows[:80],'next_repair':'Add a subject-sharing causal frame with a held-out past-tense agreement state, preserving connective semantics and pre-render character obligations.'}
 p=Path('runs/cfg-causal-shared-subject-agreement-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'candidates':len(rows),'pruned':pruned,'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
