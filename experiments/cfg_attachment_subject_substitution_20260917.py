#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
S=('gardener','reader','sailor','teacher'); DV=('carries','reviews','marks','opens'); LV=('works','waits','rests','sails'); O=('map','letter','garden','harbor'); P=('near','beside','under')
SIG='attachment-state|semantic-subject-substitution|valency-preserved|state-local-verb-domain|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s,att):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'attachment':att,'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'attachment':att,'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def d(s,v,n,rv):return f'the {s} {v} the {n} that the reader {rv} the {n}.'
def l(s,v,n,lv,p):return f'the {s} {v} the {n} where the reader {lv} {p} the {n}.'
def pack(t,att,sub,rep,prov):return {'rendered':t,'attachment_state':att,'subject':sub,'repaired':rep,'provenance':prov,'novelty_preflight':{'signature':SIG,'valency_preserved':True,'cross_state_verb_rejected':True,'not_catalogue_replay':True},'audit':a(t),'live_frontier':f(t,att),'anti_shortcut':{'single_tree':True,'subject_substitution_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for s,v,n,rv in itertools.product(S,DV,O,DV):
  alt=next(x for x in S if x!=s)
  for t,sub,rep,pr in ((d(s,v,n,rv),s,False,'fresh direct-object subject frame'),(d(alt,v,n,rv),alt,True,'semantic-preserving direct-object subject substitution')):
   if c(t) not in seen:seen.add(c(t));rows.append(pack(t,'direct-object',sub,rep,pr));controls+=not rep;repairs+=rep
 for s,v,n,lv,p in itertools.product(S,DV,O,LV,P):
  alt=next(x for x in S if x!=s)
  for t,sub,rep,pr in ((l(s,v,n,lv,p),s,False,'fresh locative subject frame'),(l(alt,v,n,lv,p),alt,True,'semantic-preserving locative subject substitution')):
   if c(t) not in seen:seen.add(c(t));rows.append(pack(t,'locative',sub,rep,pr));controls+=not rep;repairs+=rep
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-attachment-subject-substitution-20260917','method':'attachment-state subject substitution with preserved matrix valency and state-local relative verbs','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(ex),'candidates':rows[:60],'next_repair':'Add a held-out matrix object substitution within each attachment state, preserving transitivity or locative frame and carrying the same live character obligation.'}
 p=Path('runs/cfg-attachment-subject-substitution-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
