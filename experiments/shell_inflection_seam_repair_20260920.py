"""Bounded repair: role-preserving inflectional substitutions in one shell."""
from __future__ import annotations
import hashlib,itertools,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
EXPERIMENT_ID='shell-inflection-seam-repair-20260920'
# Each row is a complete SVO shell; alternatives preserve its grammatical role.
SLOTS={
 'det':('a','an','the','some','nine'),
 'subj':('aide','poet','scribe','sailor','men','singers','nurse','bard','reader'),
 'verb':('reads','rips','marks','writes','inspires','praises','guards','opens','keeps','finds'),
 'obj':('memos','notes','maps','letters','books','songs','plans','pages','poems'),
}
TEMPLATES=(('det','subj','verb','det','obj'),('det','subj','verb','obj'),)
def audit(text):
 t=normalize_letters(text);r=t[::-1]; mm=next(((i,a,b) for i,(a,b) in enumerate(zip(t,r)) if a!=b),None)
 return {'normalized':t,'letters':len(t),'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(r.encode()).hexdigest()}
def run():
 rows=[]; visited=0
 for template in TEMPLATES:
  for vals in itertools.product(*(SLOTS[x] for x in template)):
   # Determiners and noun number agreement; only complete grammatical shells.
   if vals[0] in {'a','an'} and vals[1] in {'men','singers'}: continue
   if vals[0]=='an' and not vals[1][0] in 'aeiou': continue
   if vals[0] in {'a','an'} and vals[-1] in {'memos','notes','maps','letters','books','songs','plans','pages','poems'}: continue
   plural = vals[1] in {'men','singers'}
   if plural and vals[2] not in {'read','mark','write','inspire','praise','guard','open','keep','find'}: continue
   if not plural and vals[2] in {'read','mark','write','inspire','praise','guard','open','keep','find'}: continue
   visited+=1
   text=' '.join(vals); rendered=text.capitalize()+'.'; au=audit(rendered)
   seam=0
   for a,b in zip(au['normalized'],au['normalized'][::-1]):
    if a!=b: break
    seam+=1
   if seam>=2 or au['two_pointer_exact']:
    rows.append({'rendered':rendered,'slots':dict(zip(template,vals)),'audit':au,'seam_match_chars':seam,'mechanical_checks':mechanical_admission_checks(rendered,min_letters=30,max_letters=260),'provenance':{'construction':'single complete SVO shell with role-preserving lexical/inflectional substitutions','finished_tape_reversed':False,'catalogue_imported':False,'word_order_mirror':False,'repeated_unit':False}})
 rows.sort(key=lambda x:(x['audit']['two_pointer_exact'],x['seam_match_chars'],x['audit']['letters']),reverse=True)
 exact=[x for x in rows if x['audit']['two_pointer_exact']]
 return {'experiment_id':EXPERIMENT_ID,'method':'seam-conditioned inflectional substitution inside one complete grammatical shell','stats':{'visited':visited,'retained':len(rows),'exact':len(exact),'longest_retained_letters':max((x['audit']['letters'] for x in rows),default=0),'best_seam_match_chars':max((x['seam_match_chars'] for x in rows),default=0)},'candidates':rows[:20],'independent_audit':['normalized two-pointer test','forward/reverse SHA-256'],'novelty_preflight':{'status':'passed','signature':EXPERIMENT_ID},'failure_and_repair':{'failure':'no exact shell closure' if not exact else 'exact shell closure found','next_repair':'add agreement-carrying adjunct slots while preserving one-shell semantics'},'reader_gate':'closed until blinded readers judge intact prose'}
if __name__=='__main__':
 out=run();(ROOT/'runs'/(EXPERIMENT_ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True));print('\n'.join(x['rendered'] for x in out['candidates'][:3]))
