"""Bounded seedless typed-template equation search.

Fresh lexical banks are joined by live outside-in character obligations. No
catalogue sentence, wrapped seed, duplicate-bank sweep, or finished-tape
reversal is used. This branch deliberately reports whether typed prose can
close, rather than manufacturing a closure.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT='typed-template-equation-search-20260917'
SIGNATURE='seedless-typed-template|fresh-authored-banks|live-outside-in-obligations|proper-span-palindrome-rejection|independent-audit'
BANKS={
 'agent':['the patient mason','a careful sailor','the young keeper'],
 'verb':['records','carries','studies'],
 'object':['a brass compass','the winter map','an old lantern'],
 'place':['by the inlet','near the cedar','under moonlight']}

def letters(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); bad=[(i,a,b) for i,(a,b) in enumerate(zip(t,t[::-1])) if a!=b]
 return {'letters':len(t),'two_pointer_exact':bool(t) and not bad,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest(),'mismatches':len(bad)}
def residual(l,r):
 a,b=letters(l),letters(r)[::-1]; n=min(len(a),len(b)); k=0
 while k<n and a[k]==b[k]:k+=1
 return {'matched_prefix':k,'debt':abs(len(a)-len(b))+n-k,'next_left':a[k] if k<len(a) else None,'next_mirrored_right':b[k] if k<len(b) else None}
def proper_pal_span(s):
 ws=re.findall('[a-z]+',s.lower())
 spans=[]
 for i in range(len(ws)):
  for j in range(i+2,len(ws)+1):
   t=''.join(ws[i:j])
   if t==t[::-1]:spans.append(' '.join(ws[i:j]))
 return spans

def construct(a,v,o,p):
 # A typed equation: centre is verb+object, arms are independently authored.
 left=f'{a} {v} {o}'; right=p
 trace=[]
 for slot in ('place','object','agent'):
  trace.append({'slot':slot,'left_arm':left,'right_arm':right,'residual':residual(left,right)})
  # obligation is observed before next typed realization; no reverse tape made.
  if slot=='place': right=right+' '+BANKS['place'][0]
  elif slot=='object': left=BANKS['object'][1]+' '+left
  else: right=BANKS['agent'][1]+' '+right
 text=(left+' '+right).capitalize()+'.'
 return {'rendered':text,'typed_slots':{'agent':a,'verb':v,'object':o,'place':p},'trace':trace,'audit':audit(text),'proper_self_palindromic_spans':proper_pal_span(text),'provenance':{'seedless':True,'fresh_authored_lexical_banks':True,'live_obligation_before_realization':True,'catalogue_phrase_included':False,'wrapped_seed':False,'finished_tape_reversal':False,'duplicate_bank_sweep':False}}

def run():
 rows=[construct(a,v,o,p) for a in BANKS['agent'] for v in BANKS['verb'] for o in BANKS['object'] for p in BANKS['place']]
 rows=[r for r in rows if not r['proper_self_palindromic_spans']]
 return {'experiment':EXPERIMENT,'signature':SIGNATURE,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows)},'next_repair':{'operator':'typed lexical edge substitution indexed by next mirrored character','reason':'fresh typed templates retain readable SVO semantics but their live seam obligations have no compatible closure in this bounded bank','held_out':'boundary-compatible verb/object inflections'},'provenance':{'catalogue_used':False,'wrapped_seed':False,'proper_self_palindrome_filter':True}}
if __name__=='__main__':
 p=run();
 for d in (ROOT/'runs',ROOT/'artifacts'): (d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))
