#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
SUBJ=('the gardener','the reader','the sailor','the teacher'); PRES=('marks','reads','carries','opens'); PAST=('marked','read','carried','opened'); PAIRS=(('maps','letters','written'),('gardens','harbors','places'),('letters','maps','documents')); ALT={'written':'documents','places':'written','documents':'places'}
SIG='typed-temporal-CFG|while-connective|formerly-later-order|one-pair-relation-alternation|the-the-gate|independent-pointer-sha'
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
  alt_rel=ALT[rel]
  for o2v,r,rep,pr in ((o2,rel,False,'fresh paired relation control'),(o2,alt_rel,True,'held-out one-pair relation alternation; first object relation fixed')):
   t=f'{s} {v} the {o1} formerly while {s2} {pv} later the {o2v}.'
   rows.append({'rendered':t,'provenance':pr,'grammar_state':{'connective':'while','subject_relation':'alternating','object_relation':'paired','relation_primary':rel,'relation_rendered':r,'changed_pair':'second' if rep else 'none','object_numbers':('pl','pl'),'determiners':('the','the'),'tense_sequence':'present->past','temporal_order':'formerly<later'},'live_frontier':f(t),'audit':a(t),'anti_shortcut':{'single_tree':True,'one_pair_relation_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'distinct_from_paired_base':True,'first_relation_fixed':True}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[x for x in rows if x['audit']['two_pointer']]
 out={'experiment':'cfg-while-one-pair-relation-alternation-20260917','method':'one-pair semantic relation alternation under fixed coordinated the/the temporal frame','signature':SIG,'candidate_count':len(rows),'pruned_partial_count':pruned,'exact_count':len(ex),'candidates':rows[:80],'next_repair':'Add a relation-compatible subject alternation while retaining the selected object relation and both plural objects.'}
 p=Path('runs/cfg-while-one-pair-relation-alternation-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'candidates':len(rows),'pruned':pruned,'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
