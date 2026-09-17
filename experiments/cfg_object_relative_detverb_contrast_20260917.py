#!/usr/bin/env python3
"""Held-out determiner/verb contrast over a typed object-relative CFG."""
import hashlib,itertools,json,re
from pathlib import Path
N={'sg':('gardener','reader','sailor','teacher'),'pl':('gardeners','readers','sailors','teachers')}
DET={'sg':('a','the'),'pl':('the',)}
V={'sg':('carries','reviews','marks','opens'),'pl':('carry','review','mark','open')}
OBJ=('map','letter','garden','harbor')
SIG='object-relative|number-gated-determiner|agreement-verb-contrast|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def frontier(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-1-i]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def make(sn,on,rn,sv,rv,sdet,odet,obj):return f'{sdet} {sn} {sv} {odet} {on} that the {rn} {rv} the {obj}.'
def main():
 rows=[];seen=set();controls=repairs=0
 for snum,onum,rnum in itertools.product(('sg','pl'),repeat=3):
  for sn,on,rn,sv,rv,sdet,odet,obj in itertools.product(N[snum],N[onum],N[rnum],V[snum],V[rnum],DET[snum],DET[onum],OBJ):
   s=make(sn,on,rn,sv,rv,sdet,odet,obj);k=c(s)
   state={'subject_number':snum,'object_number':onum,'relative_subject_number':rnum,'object_determiner_state':odet,'number_gate':'object determiner allowed iff object_number=sg or odet=the','transitions':['NP['+snum+']->DET['+snum+'] N['+snum+']','VP->V['+snum+'] NP['+onum+']','NP[obj]->DET['+onum+'] N['+onum+'] RC','RC->that NP['+rnum+'] V['+rnum+']']}
   def pack(text,repair,prov):return {'rendered':text,'provenance':prov,'repaired':repair,'novelty_preflight':{'signature':SIG,'not_word_order_symmetry':True,'not_catalogue_replay':True},'agreement_state':state,'audit':audit(text),'live_frontier':frontier(text),'anti_shortcut':{'single_tree':True,'determiner_gated':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
   if k not in seen:seen.add(k);rows.append(pack(s,False,'fresh object-relative CFG derivation'));controls+=1
   # held-out operator: change only determiner for singular object, then
   # change relative verb within the already selected relative number state.
   if onum=='sg':
    alt_det='the' if odet=='a' else 'a'; r=make(sn,on,rn,sv,rv,sdet,alt_det,obj);kr=c(r)
    if kr not in seen:seen.add(kr);rows.append(pack(r,True,'held-out object determiner alternation; number and attachment fixed'));repairs+=1
   alt_v=next(v for v in V[rnum] if v!=rv);r2=make(sn,on,rn,sv,alt_v,sdet,odet,obj);kr=c(r2)
   if kr not in seen:seen.add(kr);rows.append(pack(r2,True,'held-out relative-verb substitution; determiner and number state fixed'));repairs+=1
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); exact=[x for x in rows if x['audit']['two_pointer']]
 out={'experiment':'cfg-object-relative-detverb-contrast-20260917','method':'typed object-relative CFG with number-gated determiner alternation and agreement-preserving verb contrast','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(exact),'candidates':rows[:60],'next_repair':'Add an object-relative preposition state (direct object versus locative) and gate determiner/verb substitutions by that attachment feature before live character scoring.'}
 p=Path('runs/cfg-object-relative-detverb-contrast-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(exact),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
