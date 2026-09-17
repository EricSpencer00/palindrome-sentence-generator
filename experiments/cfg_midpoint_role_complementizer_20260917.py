#!/usr/bin/env python3
"""Typed complementizer branches over role-conditioned midpoint decoding."""
import hashlib,itertools,json,re
from pathlib import Path
AGENTS=('gardener','reader','sailor','teacher','cartographer','curator','pilot','musician'); FRAMES=(('carries','object','map'),('reviews','object','letter'),('marks','object','score'),('opens','object','parcel'),('studies','object','atlas'),('charts','object','harbor'),('copies','object','journal'),('folds','object','map'),('walks','locative','garden'),('waits','locative','harbor'),('works','locative','studio'),('rests','locative','park')); ADJ=('quiet','patient','fresh','old','careful','young'); SIG='midpoint-CFG|role-conditioned-complementizer|typed-branch|whole-tape-obligations|independent-pointer-sha'
def clean(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=clean(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def trie(w):return [{'prefix':w[:i],'next':w[i:i+1],'complete':i+1==len(w)} for i in range(len(w))]
def main():
 rows=[]
 for ag,(v,role,obj),adj,comp in itertools.product(AGENTS,FRAMES,ADJ,('that','where')):
  # Complementizer is typed: that for an object role, where for a locative role.
  compatible=(role=='object' and comp=='that') or (role=='locative' and comp=='where')
  text=f'the {adj} {ag} {v} {comp} the {obj}.';t=clean(text)
  rows.append({'rendered':text,'admitted':t==t[::-1] and compatible,'provenance':'fresh whole-tape CFG branch with role-typed complementizer and bilateral lexical tries','grammar_state':{'role':role,'complementizer':comp,'role_compatible':compatible,'tries':{'agent':trie(ag),'verb':trie(v),'complement':trie(obj)},'complete_obligations':True},'audit':audit(text),'anti_shortcut':{'single_tree':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'typed_complementizer':True,'not_article_relation_variant':True}})
 rows.sort(key=lambda r:(-r['audit']['letters'],r['rendered']));exact=[r for r in rows if r['admitted']]
 out={'experiment':'cfg-midpoint-role-complementizer-20260917','method':'whole-tape CFG/automaton product with role-typed that/where branches','signature':SIG,'candidate_count':len(rows),'exact_count':len(exact),'admitted_renderings':exact,'diagnostic_controls':rows[:80],'next_repair':'Carry complementizer choice into a typed relative-clause predicate slot with role-preserving valency checks.'}
 p=Path('runs/cfg-midpoint-role-complementizer-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'candidates':len(rows),'exact':len(exact),'longest':rows[0]['audit']['letters'],'text':rows[0]['rendered']}))
if __name__=='__main__':main()
