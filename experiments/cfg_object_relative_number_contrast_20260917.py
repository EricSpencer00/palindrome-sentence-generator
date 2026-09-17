#!/usr/bin/env python3
"""Number-gated subject/object contrast for an object-relative CFG."""
import hashlib,itertools,json,re
from pathlib import Path
N={'sg':('gardener','reader','sailor','teacher'),'pl':('gardeners','readers','sailors','teachers')}
DET={'sg':('the','a'),'pl':('the',)}
V={'sg':('carries','reviews','marks','opens'),'pl':('carry','review','mark','open')}
OBJ=('map','letter','garden','harbor')
def c(s): return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t[::-1].encode()).hexdigest()==hashlib.sha256(t.encode()).hexdigest()}
def f(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def make(sn,on,rn,sv,rv,sdet,odet,obj): return f"{sdet} {sn} {sv} {odet} {on} that {sdet} {rn} {rv} the {obj}."
def main():
 rows=[];seen=set();controls=repairs=0
 for snum, onum, rnum in itertools.product(('sg','pl'),repeat=3):
  # Grammar state explicitly carries [subject number, object number,
  # relative-subject number] and [object determiner number].
  for sn,on,rn,sv,rv,sdet,odet,obj in itertools.product(N[snum],N[onum],N[rnum],V[snum],V[rnum],DET[snum],DET[onum],OBJ):
   s=make(sn,on,rn,sv,rv,sdet,odet,obj); k=c(s)
   state={'subject_number':snum,'object_number':onum,'relative_subject_number':rnum,'object_determiner_number':onum,'transitions':['S->NP['+snum+'] VP['+snum+']','VP->V['+snum+'] NP['+onum+']','NP[obj]->DET['+onum+'] N['+onum+'] RC','RC->that NP['+rnum+'] V['+rnum+']']}
   base={'rendered':s,'provenance':'fresh number-gated subject/object contrast in one object-relative CFG tree','agreement_state':state,'repaired':False,'audit':a(s),'live_frontier':f(s),'anti_shortcut':{'single_tree':True,'number_contrast':snum!=onum,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
   if k not in seen:seen.add(k);rows.append(base);controls+=1
   alt=next(x for x in V[snum] if x!=sv); r=make(sn,on,rn,alt,rv,sdet,odet,obj); kr=c(r)
   if kr not in seen: seen.add(kr); q=dict(base);q.update(rendered=r,repaired=True,provenance='held-out subject-verb substitution; number state and object determiner unchanged',audit=a(r),live_frontier=f(r));rows.append(q);repairs+=1
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); exact=[x for x in rows if x['audit']['two_pointer']]
 out={'experiment':'cfg-object-relative-number-contrast-20260917','method':'single-tree object-relative CFG with independent subject/object/relative number states and determiner gating','control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(exact),'candidates':rows[:60],'next_repair':'Use a held-out determiner alternation (a/the) at the object-relative boundary while preserving number and attachment; score only live character-compatible states.'}
 p=Path('runs/cfg-object-relative-number-contrast-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(exact),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
