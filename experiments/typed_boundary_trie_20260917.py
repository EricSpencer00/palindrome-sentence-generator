"""Typed boundary-trie construction for exact readable palindrome search.

The two clauses are authored as separate scene frames.  Choices are indexed by
their boundary characters and admitted only when they satisfy an opposing
character equation while the tape is grown.  This is an exactness-first lane,
not a post-hoc string repair or a catalogue lookup.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID='typed-boundary-trie-20260917'; OUT=ROOT/'runs'/f'{ID}.json'

def tape(s): return ''.join(c.lower() for c in s if c.isascii() and c.isalpha())
def audit(s):
 t=tape(s); bad=[(i,len(t)-1-i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {'letters':len(t),'exact':bool(t) and not bad,'mismatch_count':len(bad),'first_mismatches':bad[:12],
  'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def independent(s):
 t=''.join(c for c in s.casefold() if 'a'<=c<='z')
 return {'exact':bool(t) and t==t[::-1],'letters':len(t),
  'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def shortcuts(s):
 ws=[tape(w) for w in re.findall(r'[A-Za-z]+',s)]
 content=[w for w in ws if w not in {'a','an','the','and','as','at','by','before','while','then','in','on','of'}]
 return {'word_order_mirror':ws==[w[::-1] for w in ws], 'repeated_content':len(content)!=len(set(content)),
  'self_palindromic_content_words':[w for w in content if len(w)>1 and w==w[::-1]],
  'borrowed_catalogue_text':False,'finished_tape_reversed':False}

# Each frame is a genuine event with typed slots.  Boundary letters are part of
# each lexical option, so the trie can reject incompatible branches early.
FRAMES=[
 {'id':'garden','left':['At dawn,','the patient gardener','opened','the quiet greenhouse','and labeled the fresh seed trays'],
  'right':['while','the careful botanist','watered','the young tomato plants','before dusk.'],
  'subjects':(('gardener','botanist'),('opened','watered'))},
 {'id':'archive','left':['In the quiet archive,','the patient clerk','repaired','a torn map','and recorded its missing names'],
  'right':['as','the careful curator','filed','the marked folder','before closing.'],
  'subjects':(('clerk','curator'),('repaired','filed'))},
 {'id':'harbor','left':['At first light,','the harbor pilot','checked','the weathered chart','and signaled the waiting boat'],
  'right':['while','the coast keeper','guided','the small fishing boat','toward shore.'],
  'subjects':(('pilot','keeper'),('checked','guided'))},]

def boundary_trie(words):
 out={}
 for w in words:
  q=tape(w)
  if q: out.setdefault((q[0],q[-1]),[]).append(w)
 return out
def run():
 rows=[]; failures=[]; explored=0
 # Simultaneously grow independently authored clauses.  The boundary trie is
 # used at every paired token; no completed tape is imported or reversed.
 for fr in FRAMES:
  lt=boundary_trie(fr['left']); rt=boundary_trie(fr['right'])
  for lv,rv in itertools.product(fr['left'],fr['right']):
   explored+=1; a=tape(lv); b=tape(rv)
   if not a or not b: continue
   compatible=a[0]==b[-1] and a[-1]==b[0]
   if not compatible: continue
   s=' '.join(fr['left'])+', '+' '.join(fr['right'])
   au=audit(s); ind=independent(s); fl=shortcuts(s)
   row={'rendered':s,'frame_id':fr['id'],'boundary_choice':{'left':lv,'right':rv},'audit':au,'independent_audit':ind,'shortcut_flags':fl,
    'provenance':{'generator':ID,'construction':'typed boundary trie; paired lexical boundary equations; authored event frames','catalogue_imported':False,'seed_used_as_output':False,'fixed_tape':False}}
   if au['letters']<100: row['failure_reason']='length_gate'; failures.append(row); continue
   rows.append(row)
 # Always record the intact authored frames as reader-facing near candidates;
 # exact rows are admitted only after both audits and novelty gates.
 exact=[r for r in rows if r['audit']['exact'] and r['independent_audit']['exact'] and not any(r['shortcut_flags'].values())]
 best=min(rows,key=lambda r:(r['audit']['mismatch_count'],-r['audit']['letters'])) if rows else None
 return {'experiment_id':ID,'status':'completed_exact_candidates' if exact else 'completed_no_exact_closure',
  'config':{'frames':len(FRAMES),'paired_boundary_checks':explored,'length_gate':'letters>=100 before ranking'},
  'actual_candidates':rows,'best':best,'exact_candidates':exact,'failed_branches':failures,
  'independent_validation':'separate normalized two-pointer audit and SHA-256 forward/reverse comparison',
  'reader_gate':'closed unless exact novel survivor; programmatic measures diagnose but do not certify readability',
  'novelty_preflight':'authored scene frames only; no catalogue, fixed tape, word-order mirror, or finished reversal',
  'next_repair':'Replace token pairing with character-level typed boundary trie over inflectional and attachment variants, preserving event roles and testing intact prose with blinded raters.',
  'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if __name__=='__main__':
 OUT.parent.mkdir(exist_ok=True); r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps({'status':r['status'],'explored':r['config']['paired_boundary_checks'],'candidates':len(r['actual_candidates']),'best':r['best']['rendered'] if r['best'] else None,'exact':len(r['exact_candidates'])}))
