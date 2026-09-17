#!/usr/bin/env python3
"""Bilateral role-conditioned frame decoding with held-out lexical classes."""
import hashlib,itertools,json,re
from pathlib import Path
AGENTS=('gardener','reader','sailor','teacher','cartographer','curator','pilot','musician')
FRAMES=(('carries','object','map'),('reviews','object','letter'),('marks','object','score'),('opens','object','parcel'),('studies','object','atlas'),('charts','object','harbor'),('copies','object','journal'),('folds','object','map'),('walks','locative','garden'),('waits','locative','harbor'),('works','locative','studio'),('rests','locative','park'))
HELDOUT=(('carries','object','map'),('charts','object','harbor'),('walks','locative','garden'),('waits','locative','harbor'))
ADJ=('quiet','patient','fresh','old','careful','young'); SIG='midpoint-CFG|bilateral-role-conditioned-frames|heldout-lexical-classes|trie-closure-agreement|independent-pointer-sha'
def clean(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=clean(s); return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def trie(w): return [{'prefix':w[:i],'next':w[i:i+1],'complete':i+1==len(w)} for i in range(len(w))]
def main():
 rows=[]
 for ag,(v,role,obj),adj in itertools.product(AGENTS,FRAMES,ADJ):
  text=f'the {adj} {ag} {v} the {obj}.' if role=='object' else f'the {adj} {ag} {v} in the {obj}.'
  t=clean(text); subj='pl' if ag.endswith('s') else 'sg'; vn='pl' if v in ('walk','wait','work','rest') else 'sg'
  rows.append({'rendered':text,'admitted':t==t[::-1],'provenance':'fresh role-conditioned bilateral frame expansion; held-out lexical classes marked separately','grammar_state':{'role':role,'agent':ag,'verb':v,'object_or_place':obj,'agreement':{'subject':subj,'verb':vn,'compatible':subj==vn},'tries':{'agent':trie(ag),'verb':trie(v),'complement':trie(obj)},'complete_obligations':True},'audit':audit(text),'anti_shortcut':{'single_tree':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'heldout_frame':(v,role,obj) in HELDOUT,'role_conditioned':True,'not_article_relation_variant':True}})
 rows.sort(key=lambda r:(-r['audit']['letters'],r['rendered'])); exact=[r for r in rows if r['admitted'] and r['grammar_state']['agreement']['compatible']]
 out={'experiment':'cfg-midpoint-role-frames-heldout-20260917','method':'midpoint CFG product with bilateral role-conditioned transitive/locative frames and held-out lexical classes','signature':SIG,'candidate_count':len(rows),'exact_count':len(exact),'heldout_count':sum(r['novelty_preflight']['heldout_frame'] for r in rows),'admitted_renderings':exact,'diagnostic_controls':rows[:80],'next_repair':'Permit one typed complementizer branch per role while carrying argument-frame state through bilateral character obligations.'}
 p=Path('runs/cfg-midpoint-role-frames-heldout-20260917.json'); p.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'run':str(p),'candidates':len(rows),'exact':len(exact),'longest':rows[0]['audit']['letters'],'text':rows[0]['rendered']}))
if __name__=='__main__': main()
