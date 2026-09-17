#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
SUBJ=('the gardener','the reader','the sailor','the teacher'); PRES=('marks','reads','carries','opens'); PAST=('marked','read','carried','opened'); PL=('maps','letters','gardens','harbors')
SIG='typed-temporal-CFG|while-connective|formerly-later-order|plural-object-lexical-substitution|the-the-gate|independent-pointer-sha'
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
 for s,v,o,s2,pv,o2 in itertools.product(SUBJ,PRES,PL,SUBJ,PAST,PL):
  if s==s2 or o==o2:pruned+=1;continue
  alt=next(z for z in PL if z!=o)
  for t,obj,rep,pr in ((f'{s} {v} the {o} formerly while {s2} {pv} later the {o2}.',o,False,'fresh plural the/the control'),(f'{s} {v} the {alt} formerly while {s2} {pv} later the {o2}.',alt,True,'held-out first plural-object lexical substitution; second fixed')):
   rows.append({'rendered':t,'provenance':pr,'grammar_state':{'connective':'while','subject_relation':'alternating','object_numbers':('pl','pl'),'objects':(obj,o2),'determiners':('the','the'),'changed_position':'first' if rep else 'none','tense_sequence':'present->past','temporal_order':'formerly<later'},'live_frontier':f(t),'audit':a(t),'anti_shortcut':{'single_tree':True,'one_plural_object_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'distinct_from_plural_base':True,'other_object_fixed':True}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[x for x in rows if x['audit']['two_pointer']]
 out={'experiment':'cfg-while-plural-object-lexical-substitution-20260917','method':'one-position plural-object lexical substitution under coordinated the/the gate','signature':SIG,'candidate_count':len(rows),'pruned_partial_count':pruned,'exact_count':len(ex),'candidates':rows[:80],'next_repair':'Apply the analogous one-position substitution to the second plural object while preserving the first.'}
 p=Path('runs/cfg-while-plural-object-lexical-substitution-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'candidates':len(rows),'pruned':pruned,'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
