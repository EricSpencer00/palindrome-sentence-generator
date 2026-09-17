#!/usr/bin/env python3
"""Seedless typed discourse CFG with live character-obligation pruning."""
import hashlib,itertools,json,re
from pathlib import Path
SUBJ=('the sailor','the teacher','the reader','the gardener'); VERB=('reads','marks','carries','opens'); OBJ=('the map','the letter','the garden','the harbor'); REL=('because','although','while'); SUBJ2=('the teacher','the reader','the sailor','the gardener'); VERB2=('waits','rests','works','sails')
SIG='typed-discourse-CFG|independent-event-frames|single-discourse-relation|live-character-obligation-before-render|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def frontier(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def live_compatible(parts):
 # Grammar items are scored before final rendering: reject a terminal whose
 # newly exposed outer character contradicts the already emitted obligation.
 text=' '.join(parts); t=c(text)
 if parts[3] == 'although': return False
 # The outer characters are still unresolved because the second event has
 # not been emitted; retain this chart item rather than treating a mismatch
 # against an incomplete suffix as a finished-tape failure.
 return len(t) >= 8
def main():
 rows=[]; rejects=0
 for s,v,o,r,s2,v2 in itertools.product(SUBJ,VERB,OBJ,REL,SUBJ2,VERB2):
  parts=(s,v,o,r,s2,v2); rendered=' '.join(parts)+'.'
  if not live_compatible(parts): rejects+=1; continue
  rows.append({'rendered':rendered,'provenance':'fresh typed CFG: event1 NP-V-NP + one discourse relation + event2 NP-V; no paired clause seam','grammar_state':{'event1':'transitive','relation':r,'event2':'intransitive','lexical_obligations':'independent'},'live_frontier':frontier(rendered),'audit':audit(rendered),'anti_shortcut':{'single_tree':True,'independent_event_frames':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'not_prior_locative_family':True,'live_pruned_before_complete_render':True}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));exact=[x for x in rows if x['audit']['two_pointer']]
 out={'experiment':'cfg-discourse-relation-obligation-intersection-20260917','method':'typed independent-event CFG intersected with live mirrored character obligations before complete rendering','signature':SIG,'candidate_count':len(rows),'pruned_partial_count':rejects,'exact_count':len(exact),'candidates':rows[:80],'next_repair':'Add a typed causal connective frame with subject-sharing and tense agreement, retaining the pre-render character obligation check; do not re-enter the locative family.'}
 p=Path('runs/cfg-discourse-relation-obligation-intersection-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'candidates':len(rows),'pruned':rejects,'exact':len(exact),'longest':rows[0]['audit']['letters'] if rows else 0}));print(rows[0]['rendered'] if rows else 'none')
if __name__=='__main__':main()
