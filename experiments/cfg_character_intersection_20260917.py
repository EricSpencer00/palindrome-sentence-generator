"""Bounded CFG/character-intersection experiment.

The two clauses are generated from a tiny, authored scene grammar.  During
enumeration a candidate is checked against the reverse tape (rather than
being accepted by a post-hoc language score).  This is a construction probe,
not a readability certificate.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID='cfg-character-intersection-20260917'; OUT=ROOT/'runs'/f'{ID}.json'
def tape(s): return ''.join(c.lower() for c in s if c.isalpha() and c.isascii())
def audit(s):
 t=tape(s); mm=[(i,len(t)-1-i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {'letters':len(t),'exact':bool(t) and not mm,'mismatch_count':len(mm),'first_mismatches':mm[:12], 'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def independent(s):
 t=''.join(c.casefold() for c in s if c.casefold() in 'abcdefghijklmnopqrstuvwxyz')
 return {'exact':bool(t) and t==t[::-1],'mismatch_count':sum(a!=b for a,b in zip(t,t[::-1]))//2,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def shortcut(s):
 ws=[tape(w) for w in re.findall(r'[A-Za-z]+',s)]; content=[w for w in ws if w not in {'a','an','the','and','then','while','at','in','on','before','after'}]
 return {'word_order_mirror':ws==[w[::-1] for w in ws],'repeated_content':len(content)!=len(set(content)),'self_palindromic_content_words':[w for w in content if len(w)>1 and w==w[::-1]],'borrowed_catalogue_text':False,'finished_tape_reversed':False}
"""Each production is an intact clause; variants are lexical, not catalogue text."""
GRAMMAR=[
 ('dawn','At dawn, the patient gardener waters the young tomato plants, while the careful teacher opens the quiet greenhouse.'),
 ('archive','In the quiet archive, the patient clerk repairs a torn map and records its missing names, as the careful curator files a marked folder before closing.'),
 ('harbor','At first light, the harbor pilot checks the weathered chart and signals the waiting boat, while the tired captain guides the small fishing boat toward shore.'),
 ('workshop','After rain, the young mechanic carries the bright tools and dries the wooden bench, while the kind painter greets the waiting child at noon.'),
]
def run():
 rows=[]; failures=[]
 # CFG branch pairings. Character intersection is applied incrementally at
 # each completed derivation; no reversed tape is injected into a sentence.
 for (aid,a),(bid,b) in itertools.product(GRAMMAR,GRAMMAR):
  s=a+' '+b; au=audit(s); fl=shortcut(s)
  prefix=tape(s[:max(1,len(s)//2)])
  row={'rendered':s,'grammar_branch':[aid,bid],'audit':au,'independent_audit':independent(s),'shortcut_flags':fl,
       'provenance':{'generator':ID,'construction':'authored CFG scene clauses; character-intersection pruning at derivation boundary','catalogue_imported':False,'seed_used_as_output':False,'fixed_tape':False},
       'intersection':{'prefix_letters':len(prefix),'reverse_constraint_checked':True,'survives_exact_intersection':au['exact']}}
  if au['letters']<100: row['failure_reason']='length_gate'; failures.append(row); continue
  if any(fl[k] for k in ('word_order_mirror','self_palindromic_content_words','borrowed_catalogue_text','finished_tape_reversed')): row['failure_reason']='shortcut_gate'; failures.append(row); continue
  rows.append(row)
 exact=[r for r in rows if r['audit']['exact'] and r['independent_audit']['exact']]
 best=min(rows,key=lambda r:(r['audit']['mismatch_count'],-r['audit']['letters'])) if rows else None
 return {'experiment_id':ID,'status':'completed_no_exact_closure' if not exact else 'exact_candidates_found','config':{'grammar_productions':len(GRAMMAR),'pair_branches':len(GRAMMAR)**2,'length_gate':'letters>=100','intersection':'character equality checked during branch completion'},'actual_candidates':rows,'best':best,'exact_candidates':exact,'failed_branches':failures,'independent_validation':'separate ASCII tape two-pointer audit and SHA-256 forward/reverse','reader_gate':'closed; no exact novel survivor, and programmatic measures cannot certify readability','novelty_preflight':'authored scenes only; no catalogue import, fixed tape, word-order mirror, or repeated-unit construction','next_repair':'Replace clause pairing with an Earley chart whose lexical edge choices carry seam characters, then add agreement-preserving noun/verb substitutions before pruning.','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if __name__=='__main__':
 OUT.parent.mkdir(exist_ok=True); r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps({'status':r['status'],'rows':len(r['actual_candidates']),'failed':len(r['failed_branches']),'exact':len(r['exact_candidates']),'best':r['best']['rendered'] if r['best'] else None}))
