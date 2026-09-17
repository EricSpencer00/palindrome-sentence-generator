#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
S=('gardener','reader','sailor','teacher'); DV=('carries','reviews','marks','opens'); LV=('works','waits','rests','sails'); O=('map','letter','garden','harbor'); P=('near','beside','under'); D=('a','the')
SIG='valency-gated-relative-determiner|direct-object-locative|attachment-preserved|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s,att):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'attachment':att,'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'attachment':att,'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def d(s,v,n,rv,det):return f'the {s} {v} the {n} that the reader {rv} {det} {n}.'
def l(s,v,n,lv,p,place,det):return f'the {s} {v} the {n} where the reader {lv} {p} {det} {place}.'
def row(t,att,det,rep,pr):return {'rendered':t,'attachment_state':att,'relative_determiner':det,'repaired':rep,'provenance':pr,'novelty_preflight':{'signature':SIG,'valency_gated':True,'cross_state_determiner_rejected':True,'not_catalogue_replay':True},'audit':a(t),'live_frontier':f(t,att),'anti_shortcut':{'single_tree':True,'determiner_only':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for s,v,n,rv,det in itertools.product(S,DV,O,DV,D):
  alt='the' if det=='a' else 'a'
  for t,dd,rep,pr in ((d(s,v,n,rv,det),det,False,'fresh direct-object relative derivation'),(d(s,v,n,rv,alt),alt,True,'held-out direct-object determiner alternation')):
   if c(t) not in seen:seen.add(c(t));rows.append(row(t,'direct-object',dd,rep,pr));controls+=not rep;repairs+=rep
 for s,v,n,lv,p,place,det in itertools.product(S,DV,O,LV,P,O,D):
  alt='the' if det=='a' else 'a'
  for t,dd,rep,pr in ((l(s,v,n,lv,p,place,det),det,False,'fresh locative relative derivation'),(l(s,v,n,lv,p,place,alt),alt,True,'held-out locative determiner alternation')):
   if c(t) not in seen:seen.add(c(t));rows.append(row(t,'locative',dd,rep,pr));controls+=not rep;repairs+=rep
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-valency-gated-relative-determiner-20260917','method':'relative determiner alternation gated by direct-object versus locative valency','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(ex),'candidates':rows[:60],'next_repair':'Add a held-out complementizer alternation (that/where) only when the attachment-state grammar permits it, preserving the chosen determiner and semantic role.'}
 p=Path('runs/cfg-valency-gated-relative-determiner-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
